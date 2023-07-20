from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from enum import Enum

import numpy as np
import pandas as pd
import tqdm
from pandas import DataFrame
from sklearn import compose, impute, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import ugvc.utils.misc_utils as utils
from ugvc import logger
from ugvc.utils.stats_utils import get_f1, get_precision, get_recall, precision_recall_curve

# logger = logging.getLogger(__name__)


class VcfType(Enum):
    SINGLE_SAMPLE = 1
    JOINT = 2


class SingleModel:
    # pylint: disable=too-few-public-methods
    def __init__(self, threshold_dict: dict, is_greater_then: dict):
        self.threshold_dict = threshold_dict
        self.is_greater_then = is_greater_then

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_vec = np.ones(df.shape[0], dtype=bool)
        for var in self.threshold_dict:
            result_vec = result_vec & ((df[var] > self.threshold_dict[var]) == self.is_greater_then[var])
        return np.where(np.array(result_vec), "tp", "fp")


class SingleRegressionModel:
    # pylint: disable=too-few-public-methods
    def __init__(self, threshold_dict: dict, is_greater_then: dict, score_translation: list):
        self.threshold_dict = threshold_dict
        self.is_greater_then = is_greater_then
        self.score = score_translation

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        results = []
        for var in self.threshold_dict:
            result_v = (
                np.array(df[var])[:, np.newaxis] > self.threshold_dict[var][np.newaxis, :]
            ) == self.is_greater_then[var]
            results.append(result_v)
        result_vec = np.all(results, axis=0)
        scores = result_vec.mean(axis=1)
        # hack that the threshold will be compatible with the RF model proba output
        return pd.DataFrame([scores, scores]).T.to_numpy()


class SingleTrivialClassifierModel:
    def __init__(self, ignored_filters: Iterable[str] = ()):
        self.ignored_filters = set(ignored_filters).union({"PASS"})

    def predict(self, df: pd.DataFrame) -> np.array:
        def pass_filter(filter_str):
            return all(_filter in self.ignored_filters for _filter in filter_str.split(";"))

        pfiltered = df["filter"].apply(pass_filter)
        return np.where(np.array(pfiltered), "tp", "fp")

    @staticmethod
    def predict_proba(df: pd.DataFrame) -> np.array:
        res = np.array(df.tree_score.fillna(0))
        res_comp = 1 - res
        return np.vstack((res_comp, res)).T


class MaskedHierarchicalModel:
    # pylint: disable=too-few-public-methods
    BLOCK_SIZE = 1000000

    def __init__(
        self,
        _name: str,
        _group_column: str,
        _models_dict: dict,
        transformer: compose.ColumnTransformer | None = None,
        tree_score_fpr=None,
        threshold=None,
    ):
        self.name = _name
        self.group_column = _group_column
        self.models = _models_dict
        self.transformer = transformer
        self.tree_score_fpr = tree_score_fpr
        self.threshold = threshold

    def predict(self, df: pd.DataFrame, mask_column: str = None, get_numbers=False) -> pd.Series:
        """
        Makes prediction on the dataframe, optionally ignoring false-negative calls

        Parameters
        ----------
        df: pd.DataFrame
            Input dataframe
        mask_column: str
            or None (default)
            Column to look at to determine if the variant is false-negative
        get_numbers: bool
            While predicting, we can get the probability score (True) or get the classification (False)
            by applying the threshold model on these scores.

        Returns
        -------
        pd.Series
            Series of the size - number of rows in df, that contains tp/fp according to the model
        """
        if mask_column is not None:
            mask = df[mask_column] == "fn"
        else:
            mask = np.zeros(df.shape[0], dtype=bool)

        apply_df = df[~mask]
        groups = set(df[self.group_column])
        gvecs = [df[self.group_column] == g for g in groups]
        result = pd.Series(["fn"] * df.shape[0], index=df.index)
        for i, group_val in enumerate(groups):
            threshold = self.threshold
            result[(~mask) & (gvecs[i])] = self._predict_by_blocks(
                self.models[group_val],
                apply_df[apply_df[self.group_column] == group_val],
                get_numbers=get_numbers,
                threshold=threshold[group_val] if (threshold is not None) else threshold,
            )
        return result

    @staticmethod
    def _adjusted_classes(y_scores, t_val):
        return ["tp" if y >= t_val else "fp" for y in y_scores]

    def _predict_by_blocks(self, model, df, get_numbers=False, threshold=0):
        predictions = []
        for i in range(0, df.shape[0], MaskedHierarchicalModel.BLOCK_SIZE):
            # This function can be called when the query is classification (get_numbers = False )
            # or when the query is class probability (get_numbers=True)
            # In the first case if the model has a threshold, we call "predict_proba" function and
            # apply the threshold. Otherwise, we call a "predict" function
            if not get_numbers and threshold is None:  # model with no threshold that just returns class
                predict_fcn = model.predict
            else:
                predict_fcn = model.predict_proba

            if self.transformer is not None:
                predictions.append(
                    predict_fcn(self.transformer.fit_transform(df.iloc[i : i + MaskedHierarchicalModel.BLOCK_SIZE, :]))
                )
            else:
                predictions.append(predict_fcn(df.iloc[i : i + MaskedHierarchicalModel.BLOCK_SIZE, :]))
        if not get_numbers and threshold is not None:
            predictions = self._adjusted_classes(np.vstack(predictions)[:, 1], threshold)
        elif not get_numbers and threshold is None:
            predictions = np.hstack(predictions)
        elif get_numbers:
            predictions = np.vstack(predictions)[:, 1]

        return predictions


def train_threshold_models(
    concordance: pd.DataFrame,
    vtype: VcfType,
    interval_size: int | None = None,
    classify_column: str = "classify",
    annots: list = None,
) -> tuple[MaskedHierarchicalModel, pd.DataFrame]:
    """Trains threshold classifier and regressor

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    interval_size: int
        number of bases in the interval on which the labeling exists
    classify_column: str
        Classification column
    annots: list
        Annotation interval names
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"

    Returns
    -------
    tuple
        Classifier model and concordance
    """
    if annots is None:
        annots = []
    train_selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(concordance, train_selection_functions, "group")
    concordance = add_testing_train_split_column(concordance, "group", "test_train_split", classify_column)
    logger.debug(
        "******Minimal test loc: %i",
        np.nonzero(np.array(concordance["test_train_split"]))[0].min(),
    )
    logger.debug(
        "******Maximal test loc: %i",
        np.nonzero(np.array(concordance["test_train_split"]))[0].max(),
    )
    logger.debug(
        "******Average test loc: %i",
        np.nonzero(np.array(concordance["test_train_split"]))[0].mean(),
    )

    transformer = feature_prepare(output_df=True, annots=annots, vtype=vtype)
    transformer.fit(concordance)
    groups = set(concordance["group"])
    classifier_models = {}
    fpr_values = {}
    thresholds = {}
    for g_val in groups:
        (classifier_models[g_val], fpr_values[g_val], thresholds[g_val],) = train_threshold_model(
            concordance=concordance,
            test_train_split=concordance["test_train_split"],
            selection=concordance["group"] == g_val,
            gtr_column=classify_column,
            transformer=transformer,
            interval_size=interval_size,
            annots=annots,
            vtype=vtype,
        )

    return (
        MaskedHierarchicalModel(
            "Threshold classifier",
            "group",
            classifier_models,
            transformer=transformer,
            tree_score_fpr=fpr_values,
            threshold=thresholds,
        ),
        concordance,
    )


def train_threshold_model(
    concordance: pd.DataFrame,
    test_train_split: pd.Series,
    selection: pd.Series,
    gtr_column: str,
    transformer: compose.ColumnTransformer,
    vtype: VcfType,
    interval_size: int,
    annots: list = None,
) -> tuple:
    """Trains threshold regressor and classifier models

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    test_train_split: np.ndarray
        Test train split column.
    selection : pd.Series
        Boolean - rows of concordance that belong to the group being trained
    gtr_column: str
        Ground truth column
    transformer: sklearn_pandas.DataFrameMapper
        Feature mapper
    interval_size: int
        number of bases in the interval
    annots: list
        annotations added tp features
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"
    """
    if annots is None:
        annots = []
    features, _, qual_column = modify_features_based_on_vcf_type(vtype)

    quals = np.linspace(0, 500, 49)
    sors = np.linspace(0, 10, 49)

    pairs_qual_sor_threshold = [(quals[i], sors[j]) for i in range(len(quals)) for j in range(len(sors))]

    fns = np.array(concordance[gtr_column] == "fn")
    train_data = concordance[selection & (~fns) & test_train_split][features + annots]

    train_data = transformer.transform(train_data)
    _validate_data(train_data.to_numpy())
    labels = concordance[selection & (~fns) & test_train_split][gtr_column]
    _validate_data(labels.to_numpy())
    enclabels = np.array(labels == "tp")
    train_qual = train_data[qual_column]
    train_sor = train_data["sor__sor"]

    qq_val = train_qual.to_numpy()[:, np.newaxis] > quals[np.newaxis, :]
    ss_val = train_sor.to_numpy()[:, np.newaxis] < sors[np.newaxis, :]
    predictions_tp = qq_val[..., np.newaxis] & ss_val[:, np.newaxis, :]
    tps = (predictions_tp & enclabels[:, np.newaxis, np.newaxis]).sum(axis=0)
    fns = ((~predictions_tp) & enclabels[:, np.newaxis, np.newaxis]).sum(axis=0)
    fps = (predictions_tp & (~enclabels[:, np.newaxis, np.newaxis])).sum(axis=0)

    recalls = tps / (tps + fns)
    precisions = (tps + 1) / (tps + fps + 1)
    results_df = pd.DataFrame(
        data=np.vstack((recalls.flat, precisions.flat)).T,
        index=pairs_qual_sor_threshold,
        columns=[("recall", "var"), ("precision", "var")],
    )

    f1_val = (
        2
        * results_df[("recall", "var")]
        * results_df[("precision", "var")]
        / (results_df[("recall", "var")] + results_df[("precision", "var")])
    )
    results_df["f1"] = f1_val
    best = results_df["f1"].idxmax()

    rsi = get_r_s_i(results_df, "var")[-1].copy()
    rsi.sort_values(("precision", "var"), inplace=True)
    rsi["score"] = np.linspace(0, 1, rsi.shape[0])
    regression_model = SingleRegressionModel(
        {
            qual_column: np.array([x[0] for x in rsi.index]),
            "sor__sor": np.array([x[1] for x in rsi.index]),
        },
        {"sor__sor": False, qual_column: True},
        np.array(rsi["score"]),
    )
    tree_scores = regression_model.predict_proba(train_data)[:, 1]
    tree_scores_sorted, fpr_values = fpr_tree_score_mapping(
        tree_scores, labels, test_train_split[selection], interval_size
    )
    return (
        regression_model,
        pd.concat([pd.Series(tree_scores_sorted), fpr_values], axis=1),
        rsi["score"][best],
    )


def get_r_s_i(results: pd.DataFrame, var_type: pd.DataFrame) -> tuple:
    """Returns data for plotting ROC curve

    Parameters
    ----------
    results: pd.DataFrame
        Output of vcf_pipeline_utils.find_threshold
    var_type: str
        'snp' or 'indel' or anything else according to the names in the results

    Returns
    -------
    tuple: (pd.Series, pd.Series, np.array, pd.DataFrame)
        recall of the variable, precision of the variable, indices of rows
        to calculate ROC curve on (output of `max_merits`) and dataframe to plot
        ROC curve
    """

    recall = results[("recall", var_type)]
    precision = results[("precision", var_type)]
    idx = utils.max_merits(np.array(recall), np.array(precision))
    results_plot = results.iloc[idx]
    return recall, precision, idx, results_plot


def get_all_precision_recalls(results: pd.DataFrame) -> dict:
    """Return all precision-recall curves for thresholded models

    Parameters
    ----------
    results: pd.DataFrame
        Results data frame (output of find_thresholds)

    Returns
    -------
    dict:
        Dictionary with keys - groups for which different thresholds were calculated
        Values - dataframes for plotting precision recall curve, that for group g should
        be plotted as `df.plot(('recall',g), ('precision',g))
    """
    groups = {[x[1] for x in results.columns]}
    result = {}
    for g_val in groups:
        tmp = get_r_s_i(results, g_val)[-1]
        result[g_val] = np.array(np.vstack((tmp[("recall", g_val)], tmp[("precision", g_val)]))).T
    return result


def tuple_break(x):
    """Returns the first element in the tuple"""
    if isinstance(x, tuple):
        return x[0]
    return 0 if (x is None or np.isnan(x)) else x


def tuple_break_second(x):
    """Returns the second element in the tuple"""
    if isinstance(x, tuple) and len(x) > 1:
        return x[1]
    return 0 if (x is None or (isinstance(x, tuple) and len(x) < 2) or np.isnan(x)) else x


def tuple_break_third(x):
    """Returns the third element in the tuple"""
    if isinstance(x, tuple) and len(x) > 1:
        return x[2]
    return 0 if (x is None or (isinstance(x, tuple) and len(x) < 2) or np.isnan(x)) else x


def motif_encode_left(x):
    """Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    x_list = list(x)
    x_list.reverse()
    num = 0
    for c_val in x_list:
        num = 10 * num + bases.get(c_val, 0)
    return num


def motif_encode_right(x):
    """Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    x_list = list(x)
    num = 0
    for c_val in x_list:
        num = 10 * num + bases.get(c_val, 0)
    return num


# def trinucleotide_encode(q):
#     bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}

#     result = 0
#     for i in range(4):
#         result += result * 10 + bases[q[i]]
#     return result


def allele_encode(x):
    """Translate base into integer.
    In case we don't get a single base, we return zero
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4}
    return bases.get(x, 0)


def gt_encode(x):
    """Checks whether the variant is heterozygous(0) or homozygous(1)"""
    if x == (1, 1):
        return 1
    return 0


def modify_features_based_on_vcf_type(vtype: VcfType = "single_sample"):
    """Modify training features based on the type of the vcf

    Parameters
    ----------
    vtype: string
        The type of the input vcf. Either "single_sample", "joint" or "dv"

    Returns
    -------
    list of features, list of transform, column used for qual (qual or qd)

    Raises
    ------
    ValueError
        If vcf is of unrecognized type.
    """

    default_filler = impute.SimpleImputer(strategy="constant", fill_value=0)

    def tuple_encode_df(s):
        return pd.DataFrame(np.array([tuple_break(y) for y in s]).reshape((-1, 1)), index=s.index)

    def tuple_encode_doublet_df(s):
        return pd.DataFrame(np.array([y[:2] for y in s]).reshape((-1, 2)), index=s.index)

    def tuple_encode_triplet_df(s):
        return pd.DataFrame(np.array([y[:3] for y in s]).reshape((-1, 3)), index=s.index)

    def motif_encode_left_df(s):
        return pd.DataFrame(np.array([motif_encode_left(y) for y in s]).reshape((-1, 1)), index=s.index)

    def motif_encode_right_df(s):
        return pd.DataFrame(np.array([motif_encode_right(y) for y in s]).reshape((-1, 1)), index=s.index)

    tuple_filter = preprocessing.FunctionTransformer(tuple_encode_df)
    tuple_encode_doublet_df_transformer = preprocessing.FunctionTransformer(tuple_encode_doublet_df)
    tuple_encode_triplet_df_transformer = preprocessing.FunctionTransformer(tuple_encode_triplet_df)

    left_motif_filter = preprocessing.FunctionTransformer(motif_encode_left_df)

    right_motif_filter = preprocessing.FunctionTransformer(motif_encode_right_df)

    # trinucleotide_filter = preprocessing.FunctionTransformer(trinucleotide_encode)

    def allele_encode_df(s):
        return pd.DataFrame(np.array([x[:2] for x in s]), index=s.index).applymap(allele_encode)

    allele_filter = preprocessing.FunctionTransformer(allele_encode_df)

    def gt_encode_df(s):
        return pd.DataFrame(np.array([gt_encode(y) for y in s]).reshape((-1, 1)), index=s.index)

    gt_filter = preprocessing.FunctionTransformer(gt_encode_df)

    features = [
        "qd",
        "sor",
        "an",
        "baseqranksum",
        "dp",
        "excesshet",
        "fs",
        "mq",
        "mqranksum",
        "readposranksum",
        "alleles",
        "indel",
        "indel_length",
        "x_gcc",
        "x_css",
        "x_lm",
        "x_rm",
        "x_hil",
        "x_hin",
        "x_il",
    ]

    transform_list = [
        ("sor", default_filler, ["sor"]),
        ("dp", default_filler, ["dp"]),
        ("alleles", allele_filter, "alleles"),
        #        ("x_hin", make_pipeline(tuple_filter, allele_filter), "x_hin"),
        #        ("x_hil", tuple_filter, ["x_hil"]),
        #        ("x_il", make_pipeline(tuple_filter, default_filler), ["x_il"]),
        ("indel", "passthrough", ["indel"]),
        # ("x_ic", default_filler),
        ("x_lm", left_motif_filter, "x_lm"),
        ("x_rm", right_motif_filter, "x_rm"),
        ("x_css", make_pipeline(tuple_filter, preprocessing.OrdinalEncoder()), "x_css"),
        ("x_gcc", default_filler, ["x_gcc"]),
    ]

    if vtype == "dv":
        qual_column = "qual"
        features.extend(["vaf", "ad"])
        transform_list.extend(
            [
                ("vaf", tuple_filter, "vaf"),
                ("ad", tuple_encode_doublet_df_transformer, "ad"),
                # ("trinuc", trinucleotide_filter,"trinuc"),
                ("mq0_ref", default_filler, ["mq0_ref"]),
                ("mq0_alt", default_filler, ["mq0_alt"]),
                ("ls_ref", default_filler, ["ls_ref"]),
                ("ls_alt", default_filler, ["ls_alt"]),
                ("rs_ref", default_filler, ["rs_ref"]),
                ("rs_alt", default_filler, ["rs_alt"]),
                ("mean_nm_ref", default_filler, ["mean_nm_ref"]),
                ("median_nm_ref", default_filler, ["median_nm_ref"]),
                ("mean_nm_alt", default_filler, ["mean_nm_alt"]),
                ("median_nm_alt", default_filler, ["median_nm_alt"]),
                ("mean_mis_ref", default_filler, ["mean_mis_ref"]),
                ("median_mis_ref", default_filler, ["median_mis_ref"]),
                ("mean_mis_alt", default_filler, ["mean_mis_alt"]),
                ("median_mis_alt", default_filler, ["median_mis_alt"]),
                ("qual", "passthrough", ["qual"]),
            ]
        )
    if vtype == "single_sample":
        qual_column = "qual"
        features.extend(
            ["qual", "ps", "ac", "ad", "gt", "xc", "gq", "pl", "af", "mleac", "mleaf", "mq0c", "scl", "scr", "hapcomp"]
        )
        transform_list.extend(
            [
                ("qual", "passthrough", ["qual"]),
                ("fs", default_filler, ["fs"]),
                ("qd", default_filler, ["qd"]),
                ("mq", default_filler, ["mq"]),
                ("an", default_filler, ["an"]),
                ("baseqranksum", default_filler, ["baseqranksum"]),
                ("excesshet", default_filler, ["excesshet"]),
                ("mqranksum", default_filler, ["mqranksum"]),
                ("readposranksum", default_filler, ["readposranksum"]),
                ("ps", default_filler, ["ps"]),
                ("ac", tuple_filter, ["ac"]),
                ("ad", tuple_filter, ["ad"]),
                ("gt", gt_filter, ["gt"]),
                ("xc", default_filler, ["xc"]),
                ("gq", default_filler, ["gq"]),
                ("pl", tuple_encode_triplet_df_transformer, ["pl"]),
                ("af", tuple_filter, ["af"]),
                ("mleac", tuple_filter, ["mleac"]),
                ("mleaf", tuple_filter, ["mleaf"]),
                ("hapcomp", tuple_filter, ["hapcomp"]),
                ("mq0c", tuple_encode_doublet_df_transformer, "mq0c"),
                ("scl", tuple_encode_doublet_df_transformer, "scl"),
                ("scr", tuple_encode_doublet_df_transformer, "scr"),
            ]
        )
    elif vtype == "joint":
        qual_column = "qd"
    elif vtype == "dv":
        qual_column = "qual"
    else:
        raise ValueError("Unrecognized VCF type")
    qual_column = "qual"
    return [features, transform_list, f"{qual_column}__{qual_column}"]


def feature_prepare(vtype: VcfType, output_df: bool = False, annots: list = None) -> compose.ColumnTransformer:
    """Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    vtype: VcfType
        The type of the input vcf. Either "single_sample" or "joint"
    output_df: bool
        Should the transformer output dataframe (for threshold models) or numpy array (for trees)
    annots: list, optional
        List of annotation features (will be transformed with "None")

    Returns
    -------
    compose.ColumnTransformer
        Transformer of the dataframe to dataframe
    """
    if annots is None:
        annots = []
    _, transform_list, _ = modify_features_based_on_vcf_type(vtype)

    for annot in annots:
        transform_list.append((annot, "passthrough", [annot]))
    if output_df:
        transformer = compose.ColumnTransformer(transform_list).set_output(transform="pandas")
    else:
        transformer = compose.ColumnTransformer(transform_list)
        transform_list.append((annot, "passthrough", [annot]))
    if output_df:
        transformer = compose.ColumnTransformer(transform_list).set_output(transform="pandas")
    else:
        transformer = compose.ColumnTransformer(transform_list)

    return transformer


def train_model(
    # pylint: disable=too-many-arguments
    concordance: pd.DataFrame,
    test_train_split: np.ndarray,
    selection: pd.Series,
    gtr_column: str,
    transformer: compose.ColumnTransformer,
    interval_size: int,
    classify_model,
    vtype: VcfType,
    annots: list = None,
    exome_weight: int = 1,
    exome_weight_annotation: str = None,
) -> tuple[DecisionTreeClassifier, DecisionTreeRegressor, pd.DataFrame]:
    """Trains model on a subset of dataframe that is already dividied into a testing and training set

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    test_train_split: pd.Series or np.ndarray
        Boolean array, 1 is train
    selection: pd.Series
        Boolean series that points to data selected for the model
    gtr_column: str
        Column with labeling
    transformer: sklearn_pandas.DataFrameMapper
        transformer from df -> matrix
    annots: list
        Annotation interval names
    exome_weight: int
        Weight value for the exome variants
    exome_weight_annotation: str
        Exome weight annotation name
    interval_size: int
        Number of bases in interval
    classify_model: RandomForestClassifier
        random forest classifier
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"

    Returns
    -------
    tuple:
        Trained classifier model, fpr trees-core mapping, classify threshold
    """
    if annots is None:
        annots = []

    features, _, _ = modify_features_based_on_vcf_type(vtype)

    fns = np.array(concordance[gtr_column] == "fn")
    train_data = concordance[test_train_split & selection & (~fns)][features + annots]

    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    train_data = transformer.transform(train_data)

    _validate_data(train_data)
    _validate_data(labels.to_numpy())

    model = classify_model

    if exome_weight != 1 and (exome_weight_annotation is not None) and isinstance(model, RandomForestClassifier):
        sample_weight = concordance[test_train_split & selection & (~fns)][features + annots][exome_weight_annotation]
        sample_weight = sample_weight.apply(lambda x: exome_weight if x else 1)

        model.fit(train_data, labels, sample_weight=sample_weight)
    else:
        model.fit(train_data, labels)

    tree_scores = model.predict_proba(train_data)[:, 1]
    curve = precision_recall_curve(labels, tree_scores, fn_mask=(labels == "fn"), pos_label="tp")
    precision, recall, f1_val, predictions = curve  # pylint: disable=unused-variable
    # get the best f1 threshold
    threshold = predictions[np.argmax(f1_val)]

    if gtr_column == "classify":  # there is gt
        tree_scores_sorted, fpr_values = fpr_tree_score_mapping(tree_scores, labels, test_train_split, interval_size)
        return (
            model,
            pd.concat(
                [pd.Series(tree_scores_sorted), fpr_values],
                axis=1,
            ),
            threshold,
        )
    return model, None, threshold


def train_model_rf(
    concordance: pd.DataFrame,
    test_train_split: np.ndarray,
    selection: pd.Series,
    gtr_column: str,
    transformer: compose.ColumnTransformer,
    vtype: VcfType,
    interval_size: int,
    annots: list = None,
    exome_weight: int = 1,
    exome_weight_annotation: str = None,
) -> tuple[DecisionTreeClassifier, DecisionTreeRegressor, pd.DataFrame]:
    """Trains model on a subset of dataframe that is already divided into a testing and training set

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    test_train_split: pd.Series or np.ndarray
        Boolean array, 1 is train
    selection: pd.Series
        Boolean series that points to data selected for the model
    gtr_column: str
        Column with labeling
    transformer: sklearn_pandas.DataFrameMapper
        transformer from df -> matrix
    annots: list
        Annotation interval names
    exome_weight: int
        Weight value for the exome variants
    exome_weight_annotation: str
        Exome weight annotation name
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"
    interval_size: int
        Number of bases in interval

    Returns
    -------
    Trained classifier model
    """
    if annots is None:
        annots = []
    model = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42)

    return train_model(
        concordance,
        test_train_split,
        selection,
        gtr_column,
        transformer,
        interval_size,
        model,
        annots=annots,
        exome_weight=exome_weight,
        exome_weight_annotation=exome_weight_annotation,
        vtype=vtype,
    )


def _validate_data(data: np.ndarray | pd.Series | pd.DataFrame) -> None:
    """Validates that the data does not contain nulls"""

    if type(data) == np.ndarray:  # pylint: disable=unidiomatic-typecheck
        test_data = data
    else:
        test_data = data.to_numpy()
    try:
        if len(test_data.shape) == 1 or test_data.shape[1] <= 1:
            assert pd.isnull(test_data).sum() == 0, "data vector contains null"
        else:
            for c_val in range(test_data.shape[1]):
                assert pd.isnull(test_data[:, c_val]).sum() == 0, f"Data matrix contains null in column {c_val}"
    except AssertionError as af_val:
        logger.error(str(af_val))
        raise af_val


def fpr_tree_score_mapping(
    tree_scores: np.ndarray,
    labels: pd.Series,
    test_train_split: pd.Series,
    interval_size: int,
) -> pd.Series:
    """Clclulate False Positive Rate for each variant
    '' Order the variants by incresinng order and clculate the number of false positives that we have per mega

        Parameters
        ----------
        tree_scores: pd.Series
            tree_scores values of the variants
        labels: pd.Series
            labels of tp fp fn for each variant
        test_train_split: pd.Series
            Boolean series that points to train/ test data selected for the model (true is train)
        interval_size: int
            Number of bases in interval
        Returns
        -------
        pd.Series:
            FPR value for each variant sorted in increased order
    """
    # in case we do not run frp - interval_size is None
    if interval_size is None:
        return np.zeros(len(tree_scores)), pd.Series(np.zeros(len(tree_scores)))
    train_part = sum(test_train_split) / len(test_train_split)
    tree_scores_sorted_inds = np.argsort(tree_scores)
    cur_fpr = 0
    fpr = []
    for cur_ind in tree_scores_sorted_inds[::-1]:
        if labels[cur_ind] == "fp":
            cur_fpr = cur_fpr + 1
        fpr.append((cur_fpr / train_part) / interval_size)
    return tree_scores[tree_scores_sorted_inds], pd.Series(fpr[::-1]) * 10**6


def get_basic_selection_functions():
    "Selection between SNPs and INDELs"
    sfs = []
    names = []
    sfs.append(lambda x: np.logical_not(x.indel))
    names.append("snp")
    sfs.append(lambda x: (x.indel))
    names.append("indel")

    return dict(zip(names, sfs))


def get_training_selection_functions():
    """get_training_selection_functions"""
    sfs = []
    names = []
    sfs.append(lambda x: np.logical_not(x.indel))
    names.append("snp")
    sfs.append(lambda x: (x.indel & (x.hmer_indel_length == 0)))
    names.append("non-h-indel")
    sfs.append(lambda x: (x.indel & (x.hmer_indel_length > 0)))
    names.append("h-indel")
    return dict(zip(names, sfs))


def find_thresholds(
    concordance: pd.DataFrame,
    classify_column: str = "classify",
    sf_generator: Callable = get_training_selection_functions,
) -> pd.DataFrame:
    quals = np.linspace(0, 2000, 30)
    sors = np.linspace(0, 20, 80)
    results = []
    pairs = []
    selection_functions = sf_generator()
    concordance = add_grouping_column(concordance, selection_functions, "group")
    for q_val in tqdm.tqdm_notebook(quals):
        for s_val in sors:
            pairs.append((q_val, s_val))
            tmp = (
                concordance[
                    ((concordance["qual"] > q_val) & (concordance["sor"] < s_val))
                    | (concordance[classify_column] == "fn")
                ][[classify_column, "group"]]
            ).copy()
            tmp1 = (
                concordance[
                    ((concordance["qual"] < q_val) | (concordance["sor"] > s_val))
                    & (concordance[classify_column] == "tp")
                ][[classify_column, "group"]]
            ).copy()
            tmp1[classify_column] = "fn"
            tmp2 = pd.concat((tmp, tmp1))
            results.append(tmp2.groupby([classify_column, "group"]).size())
    results_df = pd.concat(results, axis=1)
    results_df = results_df.T
    results_df.columns = results_df.columns.to_flat_index()

    for group in set(concordance["group"]):
        results_df[("recall", group)] = results_df.get(("tp", group), 0) / (
            results_df.get(("tp", group), 0) + results_df.get(("fn", group), 0) + 1
        )
        results_df[("precision", group)] = results_df.get(("tp", group), 0) / (
            results_df.get(("tp", group), 0) + results_df.get(("fp", group), 0) + 1
        )
        results_df.index = pairs
    return results_df


def add_grouping_column(df: pd.DataFrame, selection_functions: dict, column_name: str) -> pd.DataFrame:
    """
    Add a column for grouping according to the values of selection functions

    Parameters
    ----------
    df: pd.DataFrame
        concordance dataframe
    selection_functions: dict
        Dictionary of selection functions to be applied on the df, keys - are the name of the group
    column_name: str
        Name of the column to contain grouping

    Returns
    -------
    pd.DataFrame
        df with column_name added to it that is filled with the group name according
        to the selection function
    """
    df[column_name] = None
    for k in selection_functions:
        df.loc[selection_functions[k](df), column_name] = k
    return df


def tree_score_to_fpr(df: pd.DataFrame, prediction_score: pd.Series, tree_score_fpr: pd.DataFrame) -> pd.DataFrame:
    """
    Deduce frp value from the tree_score and the tree score fpr mapping

        Parameters
        ----------
        df: pd.DataFrame
            concordance dataframe
        prediction_score: pd.Series
            prediction_score
        tree_score_fpr: pd.DataFrame
            dict -> pd.DataFrame
            dictionary of group -> df were the df is
            2 columns of tree score and its corresponding fpr in increasing order of tree_score
            and the group key is snp, h-indel, non-h-indel

        Returns
        -------
        pd.DataFrame
            df with column_name added to it that is filled with the fpr value according
            to the tree score fpr mapping
    """

    fpr_values = pd.Series(np.zeros(len(prediction_score)))
    fpr_values.index = prediction_score.index
    for group in df["group"].unique():
        select = df["group"] == group
        tree_score_fpr_group = tree_score_fpr[group]
        if tree_score_fpr_group is not None:
            # it is None in case we didn't run training on gt, but on dbsnp and blacklist
            fpr_values.loc[select] = np.interp(
                prediction_score.loc[select],
                tree_score_fpr_group.iloc[:, 0],
                tree_score_fpr_group.iloc[:, 1],
            )
    return fpr_values


def get_testing_selection_functions() -> OrderedDict:
    sfs = OrderedDict()
    sfs["SNP"] = lambda x: np.logical_not(x.indel)
    sfs["Non-hmer INDEL"] = lambda x: x.indel & (x.hmer_indel_length == 0)
    sfs["HMER indel <= 4"] = lambda x: x.indel & (x.hmer_indel_length > 0) & (x.hmer_indel_length < 5)
    sfs["HMER indel (4,8)"] = lambda x: x.indel & (x.hmer_indel_length >= 5) & (x.hmer_indel_length < 8)
    sfs["HMER indel [8,10]"] = lambda x: x.indel & (x.hmer_indel_length >= 8) & (x.hmer_indel_length <= 10)
    sfs["HMER indel 11,12"] = lambda x: x.indel & (x.hmer_indel_length >= 11) & (x.hmer_indel_length <= 12)
    sfs["HMER indel > 12"] = lambda x: x.indel & (x.hmer_indel_length > 12)
    return sfs


def add_testing_train_split_column(
    concordance: pd.DataFrame,
    training_groups_column: str,
    test_train_split_column: str,
    gtr_column: str,  # pylint: disable=unused-argument
    min_test_set: int = 50,
    max_train_set: int = 200000,
    test_set_fraction: float = 0.5,
) -> pd.DataFrame:
    """Adds a column that divides each training group into a train/test set. Supports
    requirements for the minimal testing set size, maximal training test size and the fraction of test.
    The training set size of each group will be:

    train_test_size = min(group_size-min_test_size, max_train_set, group_size * (1-test_sec_frac))

    TODO: gtr_column currently has no effect and false negatives are chosen for test and train sets.
    (it is assumed that they are removed in the input)

    Parameters
    ----------
    concordance: pd.DataFrame
        Input data frame
    training_groups_column: str
        Name of the grouping column. Testing/training split will be performed separately
        for each group
    test_train_split_column: str
        Name of the splitting column
    gtr_column: str
        Name of the column that contains ground truth (will exclude fns)
    min_test_set: int
        Default - 2000, minimal size of the testing set of each variant group
    max_train_set: int
        Default - 200000, maximal size of the training set of each variant group
    test_set_fraction: float
        Default - 0.5, desired fraction of the test set

    Returns
    -------
    pd.DataFrame
        Dataframe with 0/1 in test_train_split_column, with 1 for train
    """
    groups = set(concordance[training_groups_column])

    test_train_split_vector = np.zeros(concordance.shape[0], dtype=bool)
    for g_val in groups:

        group_vector = concordance[training_groups_column] == g_val
        locations = group_vector.to_numpy().nonzero()[0]
        assert group_vector.sum() >= min_test_set, "Group size too small for training"
        train_set_size = int(
            min(
                group_vector.sum() - min_test_set,
                max_train_set,
                np.round(group_vector.sum() * (1 - test_set_fraction)),
            )
        )

        test_set_size = group_vector.sum() - train_set_size
        assert test_set_size >= min_test_set, f"Test set size too small -> test:{test_set_size}, train:{train_set_size}"
        assert (
            train_set_size <= max_train_set
        ), f"Train set size too big -> test:{test_set_size}, train:{train_set_size}"
        np.random.seed(42)  # making everything deterministic
        train_set = locations[
            np.random.choice(
                np.arange(group_vector.sum(), dtype=int),
                train_set_size,
                replace=False,
            )
        ]
        test_train_split_vector[train_set] = True

    concordance[test_train_split_column] = test_train_split_vector
    return concordance


def train_model_wrapper(
    concordance: pd.DataFrame,
    classify_column: str,
    interval_size: int,
    train_function,
    model_name: str,
    vtype: VcfType,
    annots: list = None,
    exome_weight: int = 1,
    exome_weight_annotation: str = None,
    use_train_test_split: bool = True,
) -> tuple:
    """Train a decision tree model on the dataframe

    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe
    classify_column: str
        Ground truth labels
    interval_size: int
        number of bases in the interval
    train_function: def
        The inner function to call for training
    model_name: str
        The name of the model to save
    annots: list
        Annotation interval names
    exome_weight: int
        Weight value for the exome variants
    exome_weight_annotation: str
        Exome weight annotation name
    use_train_test_split: bool
        Whether to split the data to train/test(True) or keep all the data(False)
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"


    Returns
    -------
    (MaskedHierarchicalModel, pd.DataFrame
        Models for each group, DataFrame with group for a hierarchy group and test_train_split columns

    """
    if annots is None:
        annots = []
    logger.info("Train model %s", model_name)
    train_selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(concordance, train_selection_functions, "group")
    if use_train_test_split:
        concordance = add_testing_train_split_column(concordance, "group", "test_train_split", classify_column)
    else:
        concordance["test_train_split"] = True

    logger.debug(
        "******Minimal test loc: %i",
        np.nonzero(np.array(concordance["test_train_split"]))[0].min(),
    )
    logger.debug(
        "******Maximal test loc: %i",
        np.nonzero(np.array(concordance["test_train_split"]))[0].max(),
    )
    logger.debug(
        "******Average test loc: %i",
        np.nonzero(np.array(concordance["test_train_split"]))[0].mean(),
    )

    transformer = feature_prepare(annots=annots, vtype=vtype)
    transformer.fit(concordance)
    groups = set(concordance["group"])
    classifier_models: dict = {}
    fpr_values: dict = {}
    thresholds: dict = {}
    for g_val in groups:
        classifier_models[g_val], fpr_values[g_val], thresholds[g_val] = train_function(
            concordance=concordance,
            test_train_split=concordance["test_train_split"],
            selection=concordance["group"] == g_val,
            gtr_column=classify_column,
            transformer=transformer,
            vtype=vtype,
            interval_size=interval_size,
            annots=annots,
            exome_weight=exome_weight,
            exome_weight_annotation=exome_weight_annotation,
        )

    return (
        MaskedHierarchicalModel(
            model_name + " classifier",
            "group",
            classifier_models,
            transformer=transformer,
            threshold=thresholds,
            tree_score_fpr=fpr_values,
        ),
        concordance,
    )


def eval_decision_tree_model(
    concordance: pd.DataFrame,
    model: MaskedHierarchicalModel,
    classify_column: str,
    add_testing_group_column: bool = True,
) -> DataFrame:
    """
    Calculate precision/recall for the decision tree classifier

    Parameters
    ----------
    concordance: pd.DataFrame
        Input dataframe
    model: MaskedHierarchicalModel
        Model
    classify_column: str
        Ground truth labels
    add_testing_group_column: bool
        Should default testing grouping be added (default: True),
        if False will look for grouping in group_testing

    Returns
    -------
    dict:
        Tuple dictionary - recall/precision for each category
    """

    if add_testing_group_column:
        concordance = add_grouping_column(concordance, get_testing_selection_functions(), "group_testing")
        groups = list(get_testing_selection_functions().keys())

    else:
        assert "group_testing" in concordance.columns, "group_testing column should be given"
        groups = list(set(concordance["group_testing"]))
    # get filter status as True='tp' False='fp'
    predictions = model.predict(df=concordance, mask_column=classify_column)

    accuracy_df = pd.DataFrame(
        columns=[
            "group",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "initial_tp",
            "initial_fp",
            "initial_fn",
            "initial_precision",
            "initial_recall",
            "initial_f1",
        ]
    )
    for g_val in groups:
        select = (concordance["group_testing"] == g_val) & (~concordance["test_train_split"])
        group_df = concordance[select]

        # apply filters on classification
        is_filtered: pd.Series = predictions[select] != "tp"

        post_filtering_classification = apply_filter(group_df[classify_column], is_filtered)

        pre_filtering_tp = int((group_df[classify_column] == "tp").sum())
        pre_filtering_fp = int((group_df[classify_column] == "fp").sum())
        pre_filtering_fn = int((group_df[classify_column] == "fn").sum())
        pre_filtering_precision = get_precision(pre_filtering_fp, pre_filtering_tp)
        pre_filtering_recall = get_recall(pre_filtering_fn, pre_filtering_tp)
        pre_filtering_f1 = get_f1(pre_filtering_precision, pre_filtering_recall)

        post_filtering_tp = int((post_filtering_classification == "tp").sum())
        post_filtering_fp = int((post_filtering_classification == "fp").sum())
        post_filtering_fn = int((post_filtering_classification == "fn").sum())
        post_filtering_precision = get_precision(post_filtering_fp, post_filtering_tp)
        post_filtering_recall = get_recall(post_filtering_fn, post_filtering_tp)
        post_filtering_f1 = get_f1(post_filtering_precision, post_filtering_recall)

        accuracy_df = pd.concat(
            (
                accuracy_df,
                pd.DataFrame(
                    {
                        "group": g_val,
                        "tp": post_filtering_tp,
                        "fp": post_filtering_fp,
                        "fn": post_filtering_fn,
                        "precision": post_filtering_precision,
                        "recall": post_filtering_recall,
                        "f1": post_filtering_f1,
                        "initial_tp": pre_filtering_tp,
                        "initial_fp": pre_filtering_fp,
                        "initial_fn": pre_filtering_fn,
                        "initial_precision": pre_filtering_precision,
                        "initial_recall": pre_filtering_recall,
                        "initial_f1": pre_filtering_f1,
                    },
                    index=[0],
                ),
            ),
            ignore_index=True,
        )
    return accuracy_df


def get_empty_recall_precision(category: str) -> dict:
    """Return empty recall precision dictionary for category given"""
    return {
        "group": category,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "initial_tp": 0,
        "initial_fp": 0,
        "initial_fn": 0,
        "initial_precision": 1.0,
        "initial_recall": 1.0,
        "initial_f1": 1.0,
    }


def apply_filter(pre_filtering_classification: pd.Series, is_filtered: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    pre_filtering_classification : pd.Series
        classification to 'tp', 'fp', 'fn' before applying filter
    is_filtered : pd.Series
        boolean series denoting which rows where filtered

    Returns
    -------
    pd.Series
        classification to 'tp', 'fp', 'fn', 'tn' after applying filter
    """
    post_filtering_classification = pre_filtering_classification.copy()
    post_filtering_classification.iloc[is_filtered & (post_filtering_classification == "fp")] = "tn"
    post_filtering_classification.iloc[is_filtered & (post_filtering_classification == "tp")] = "fn"
    return post_filtering_classification


def get_decision_tree_pr_curve(
    concordance: pd.DataFrame,
    model: MaskedHierarchicalModel,
    classify_column: str,
    add_testing_group_column: bool = True,
) -> pd.DataFrame:
    """
    Calculate precision/recall curve for the decision tree regressor

    Parameters
    ----------
    concordance: pd.DataFrame
        Input dataframe
    model: MaskedHierarchicalModel
        Model
    classify_column: str
        Ground truth labels
    add_testing_group_column: bool
        Should default testing grouping be added (default: True),
        if False will look for grouping in group_testing

    Returns
    -------
    pd.DataFrame:
        DataFrame of recall/precision in each category
    """

    if add_testing_group_column:
        concordance = add_grouping_column(concordance, get_testing_selection_functions(), "group_testing")
        groups = list(get_testing_selection_functions().keys())

    else:
        assert "group_testing" in concordance.columns, "group_testing column should be given"
        groups = list(set(concordance["group_testing"]))

    predictions = model.predict(concordance, classify_column, get_numbers=True)
    accuracy_df = pd.DataFrame(columns=["group", "predictions", "precision", "recall", "f1", "threshold"])

    MIN_EXAMPLE_COUNT = 20
    for g_val in groups:
        select = (concordance["group_testing"] == g_val) & (~concordance["test_train_split"])
        logger.debug(f"{g_val}: {select.sum()}")
        if select.sum() == 0:
            logger.debug(f"appending empty recall precision for {g_val}")
            accuracy_df = pd.concat(
                (accuracy_df, pd.DataFrame(pd.Series(get_empty_recall_precision_curve(g_val))).T),
            )
            continue

        classification = concordance.loc[select, classify_column]
        group_predictions = predictions[select]
        group_predictions[classification == "fn"] = -1

        group_ground_truth = classification.copy()
        group_ground_truth[classification == "fn"] = "tp"
        curve = precision_recall_curve(
            np.array(group_ground_truth),
            np.array(group_predictions),
            fn_mask=(classification == "fn"),
            pos_label="tp",
            min_class_counts_to_output=MIN_EXAMPLE_COUNT,
        )

        precision, recall, f1_val, prediction_values = curve
        if len(f1_val) > 0:
            threshold_loc = np.argmax(f1_val)
            threshold = prediction_values[threshold_loc]
        else:
            threshold = 0

        accuracy_df = pd.concat(
            (
                accuracy_df,
                pd.DataFrame(
                    pd.Series(
                        {
                            "group": g_val,
                            "predictions": prediction_values,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1_val,
                            "threshold": threshold,
                        }
                    )
                ).T,
            ),
        )

    return accuracy_df


def get_empty_recall_precision_curve(category: str) -> dict:
    """Return empty recall precision curve dictionary for category given"""
    return {"group": category, "predictions": [], "precision": [], "recall": [], "f1": [], "threshold": 0}


def calculate_unfiltered_model(concordance: pd.DataFrame, classify_column: str) -> tuple:
    """Calculates precision and recall on the unfiltered data

    Parameters
    ----------
    concordance: pd.DataFrame
        Comparison dataframe
    classify_column: str
        Classification column

    Returns
    -------
    tuple:
        MaskedHierarchicalModel, dict, model and dictionary of recalls_precisions
    """

    selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(concordance, selection_functions, "group")
    all_groups = set(concordance["group"])
    models = {}
    for g_val in all_groups:
        models[g_val] = SingleModel({}, {})
    result = MaskedHierarchicalModel("unfiltered", "group", models)
    concordance["test_train_split"] = np.zeros(concordance.shape[0], dtype=bool)
    recalls_precisions = eval_decision_tree_model(concordance, result, classify_column)
    return result, recalls_precisions


class VariantSelectionFunctions(Enum):
    # Disable all the no-member,no-self-argument violations in this function
    # pylint: disable=no-self-argument
    # pylint: disable=no-member
    # pylint: disable=invalid-name
    """Collecton of variant selection functions - all get DF as input and return boolean np.array"""

    def ALL(df: pd.DataFrame) -> np.ndarray:
        return np.ones(df.shape[0], dtype=bool)

    def HMER_INDEL(df: pd.DataFrame) -> np.ndarray:
        return np.array(df.hmer_indel_length > 0)

    def ALL_except_HMER_INDEL_greater_than_or_equal_5(df: pd.DataFrame) -> np.ndarray:
        return np.array(~((df.hmer_indel_length >= 5)))
