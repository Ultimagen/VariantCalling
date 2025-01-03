from __future__ import annotations

from collections import OrderedDict
from enum import Enum

import numpy as np
import pandas as pd
import xgboost
from pandas.core.groupby import DataFrameGroupBy
from sklearn import compose

from ugvc import logger
from ugvc.filtering import multiallelics as mu
from ugvc.filtering import transformers
from ugvc.filtering.tprep_constants import GtType, VcfType
from ugvc.utils import math_utils, stats_utils

MAX_CHUNK_SIZE = 1000000


def train_model(
    concordance: pd.DataFrame,
    gt_type: GtType,
    vtype: VcfType,
    annots: list | None = None,
) -> tuple[compose.ColumnTransformer, xgboost.XGBRFClassifier]:
    """Trains model xgboost model on a subset of dataframe

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    gt_type: GtType
        Is the ground truth approximate or exact (in the first case the model will predict 0/1, in the second: 0/1/2)
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"
    annots: list, optional
        Optional list of annotations present in the dataframe

    Returns
    -------
    tuple:
        Trained transformer and classifier model

    Raises
    ------
    ValueError
        If the gt_type is not recognized
    """
    transformer = transformers.get_transformer(vtype, annots=annots)
    if gt_type == GtType.APPROXIMATE:
        select_train = concordance["label"].apply(lambda x: x in {0, 1})
    elif gt_type == GtType.EXACT:
        select_train = concordance["label"].apply(lambda x: x in {(0, 0), (0, 1), (1, 1), (1, 0)})
    else:
        raise ValueError("Unknown gt_type")
    df_train = concordance.loc[select_train]
    if gt_type == GtType.APPROXIMATE:
        labels_train = df_train["label"].values
    elif gt_type == GtType.EXACT:
        labels_train = transformers.label_encode.transform(list(df_train["label"].values))
    else:
        raise ValueError("Unknown gt_type")

    logger.info(f"Training model on {len(df_train)} samples: start")
    x_train_df = pd.DataFrame(transformer.fit_transform(df_train))
    _validate_data(x_train_df)
    logger.info("Transform: done")
    clf = xgboost.XGBClassifier(
        n_estimators=100,
        learning_rate=0.15,
        subsample=0.4,
        max_depth=6,
        random_state=0,
        colsample_bytree=0.4,
        n_jobs=14,
    )
    clf.fit(x_train_df, labels_train)
    return clf, transformer


def apply_model(
    input_df: pd.DataFrame, model: xgboost.XGBClassifier, transformer: compose.ColumnTransformer
) -> tuple[np.ndarray, np.ndarray]:
    """Applies model to the input dataframe

    Parameters
    ----------
    input_df: pd.DataFrame
        Input dataframe
    model: xgboost.XGBClassifier
        Model
    transformer: compose.ColumnTransformer
        Transformer

    Returns
    -------
    tuple:
        Predictions and probabilities
    """

    # We apply models on batches of the dataframe to avoid memory issues
    chunks = np.arange(0, input_df.shape[0], MAX_CHUNK_SIZE, dtype=int)
    chunks = np.concatenate((chunks, [input_df.shape[0]]))
    transformed_chunks = [
        transformer.transform(input_df.iloc[chunks[i] : chunks[i + 1]]) for i in range(len(chunks) - 1)
    ]
    x_test_df = pd.concat(transformed_chunks)
    _validate_data(x_test_df)
    predictions = model.predict(x_test_df)
    probabilities = model.predict_proba(x_test_df)
    return predictions, probabilities


def _validate_data(data: np.ndarray | pd.Series | pd.DataFrame) -> None:
    """Validates that the data does not contain nulls"""

    if isinstance(data, np.ndarray):
        test_data = data
    else:
        test_data = pd.DataFrame(data).to_numpy()
    try:
        if len(test_data.shape) == 1 or test_data.shape[1] <= 1:
            assert pd.isnull(test_data).sum() == 0, "data vector contains null"
        else:
            for c_val in range(test_data.shape[1]):
                assert pd.isnull(test_data[:, c_val]).sum() == 0, f"Data matrix contains null in column {c_val}"
    except AssertionError as af_val:
        logger.error(str(af_val))
        raise af_val


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


def eval_model(
    df: pd.DataFrame,
    model: xgboost.XGBClassifier,
    transformer: compose.ColumnTransformer,
    add_testing_group_column: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate precision/recall for the decision tree classifier

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe
    model: xgboost.XGBClassifier
        Model
    transformer: compose.ColumnTransformer
        Data prep transformer
    add_testing_group_column: bool
        Should default testing grouping be added (default: True),
        if False will look for grouping in `group_testing`

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]:
        recall/precision for each category and recall precision curve for each category in dataframe

    Raises
    ------
    RuntimeError
        If the group_testing column is not present in the dataframe
    """
    df = df.copy()
    predictions, probs = apply_model(df, model, transformer)
    phred_pls = -math_utils.phred(probs)
    sorted_pls = np.sort(phred_pls, axis=1)
    gqs = sorted_pls[:, -1] - sorted_pls[:, -2]
    quals = phred_pls[:, 1:].max(axis=1) - phred_pls[:, 0]
    df["ml_gq"] = gqs
    df["ml_qual"] = quals
    df["predict"] = predictions

    labels = df["label"]
    if probs.shape[1] == 2:
        gt_type = GtType.APPROXIMATE
    elif probs.shape[1] == 3:
        gt_type = GtType.EXACT
    else:
        raise RuntimeError("Unknown gt_type")

    if gt_type == GtType.APPROXIMATE:
        select = df["label"].apply(lambda x: x in {0, 1})
    elif gt_type == GtType.EXACT:
        select = df["label"].apply(lambda x: x in {(0, 0), (0, 1), (1, 1), (1, 0)})
    labels = labels[select]
    if gt_type == GtType.EXACT:
        labels = np.array(transformers.label_encode.transform(list(labels))) > 0
        df["predict"] = df["predict"] > 0

    df = df.loc[select]
    result = evaluate_results(df, pd.Series(list(labels), index=df.index), add_testing_group_column)
    if isinstance(result, tuple):
        return result
    raise RuntimeError("Unexpected result")


def evaluate_results(
    df: pd.DataFrame,
    labels: pd.Series,
    add_testing_group_column: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate concordance results for the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    labels : pd.Series
        _description_
    add_testing_group_column : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Returns summary metrics and precision/recall curves in two dataframes
    """
    if add_testing_group_column:
        df = add_grouping_column(df, get_selection_functions(), "group_testing")
        groups = list(get_selection_functions().keys())

    else:
        assert "group_testing" in df.columns, "group_testing column should be given"
        groups = list(set(df["group_testing"]))

    accuracy_df = _init_metrics_df()
    curve_df = pd.DataFrame(columns=["group", "precision", "recall", "f1", "threshold"])
    for g_val in groups:
        select = df["group_testing"] == g_val
        group_df = df[select]
        group_labels = labels[select]
        acc, curve = get_concordance_metrics(
            group_df["predict"],
            group_df["ml_qual"],
            np.array(group_labels) > 0,
            np.zeros(group_df.shape[0], dtype=bool),
        )
        acc["group"] = g_val
        curve["group"] = g_val
        accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)
        curve_df = pd.concat((curve_df, curve), ignore_index=True)
    return accuracy_df, curve_df


def get_empty_recall_precision() -> dict:
    """Return empty recall precision dictionary for category given"""
    return {
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


def get_empty_recall_precision_curve() -> dict:
    """Return empty recall precision curve dictionary for category given"""
    return {"threshold": 0, "predictions": [], "precision": [], "recall": [], "f1": []}


def get_concordance_metrics(
    predictions: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    fn_mask: np.ndarray,
    return_metrics: bool = True,
    return_curves: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """Calculate concordance metrics. The input of predictions is assumed to be numbers,
    with zeros be negative calls. fn_mask denotes the locations that were not called in
    predictions and that are called in the truth (false negatives).
    The scores are the scores of the predictions.

    Parameters
    ----------
    predictions: np.ndarray
        Predictions (number array)
    scores: np.ndarray
        Scores (float array of scores for predictions)
    truth: np.ndarray
        Truth (number array)
    fn_mask: np.ndarray
        False negative mask (boolean array of the length of truth, predictions and scores that
        contains True for false negatives and False for the rest of the values)
    return_metrics: bool
        Convenience, should the function return metrics (True) or only precision-recall curves (False)
    return_curves: bool
        Convenience, should the function return precision-recall curves (True) or only metrics (False)

    Returns
    -------
    tuple or pd.DataFrame
        Concordance metrics and precision recall curves or one of them dependent on the return_metrics and return_curves

    Raises
    ------
    AssertionError
        At least one of return_curves or return_metrics should be True
    """

    truth_curve = truth > 0
    truth_curve[fn_mask] = True
    MIN_EXAMPLE_COUNT = 20
    precisions_curve, recalls_curve, f1_curve, thresholds_curve = stats_utils.precision_recall_curve(
        truth, scores, fn_mask, min_class_counts_to_output=MIN_EXAMPLE_COUNT
    )
    if len(f1_curve) > 0:
        threshold_loc = np.argmax(f1_curve)
        threshold = thresholds_curve[threshold_loc]
    else:
        threshold = 0

    curve_df = pd.DataFrame(
        pd.Series(
            {
                "predictions": thresholds_curve,
                "precision": precisions_curve,
                "recall": recalls_curve,
                "f1": f1_curve,
                "threshold": threshold,
            }
        )
    ).T

    fn = fn_mask.sum()
    predictions = predictions.copy()[~fn_mask]
    scores = scores.copy()[~fn_mask]
    truth = truth.copy()[~fn_mask]

    if len(predictions) == 0:

        result = (
            pd.DataFrame(get_empty_recall_precision(), index=[0]),
            pd.DataFrame(pd.Series(get_empty_recall_precision_curve())).T,
        )
    else:
        tp = ((truth > 0) & (predictions > 0) & (truth == predictions)).sum()
        fp = ((predictions > truth)).sum()
        fn = fn + ((predictions < truth)).sum()
        precision = stats_utils.get_precision(fp, tp)
        recall = stats_utils.get_recall(fn, tp)
        f1 = stats_utils.get_f1(precision, recall)
        initial_tp = (truth > 0).sum()
        initial_fp = len(truth) - initial_tp
        initial_fn = fn_mask.sum()
        initial_precision = stats_utils.get_precision(initial_fp, initial_tp)
        initial_recall = stats_utils.get_recall(initial_fn, initial_tp)
        initial_f1 = stats_utils.get_f1(initial_precision, initial_recall)
        metrics_df = pd.DataFrame(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "initial_tp": initial_tp,
                "initial_fp": initial_fp,
                "initial_fn": initial_fn,
                "initial_precision": initial_precision,
                "initial_recall": initial_recall,
                "initial_f1": initial_f1,
            },
            index=[0],
        )
        result = metrics_df, curve_df
    metrics_df, curve_df = result
    assert return_curves or return_metrics, "At least one of return_curves or return_metrics should be True"
    if return_curves and return_metrics:
        return metrics_df, curve_df
    if return_curves:
        return curve_df
    return metrics_df


def _init_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
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


def get_selection_functions() -> OrderedDict:
    sfs = OrderedDict()
    sfs["SNP"] = lambda x: np.logical_not(x.indel)
    sfs["Non-hmer INDEL"] = lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) == 0)
    sfs["HMER indel <= 4"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) > 0) & (x["x_hil"].apply(lambda y: y[0]) < 5)
    )
    sfs["HMER indel (4,8)"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) >= 5) & (x["x_hil"].apply(lambda y: y[0]) < 8)
    )
    sfs["HMER indel [8,10]"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) >= 8) & (x["x_hil"].apply(lambda y: y[0]) <= 10)
    )
    sfs["HMER indel 11,12"] = (
        lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) >= 11) & (x["x_hil"].apply(lambda y: y[0]) <= 12)
    )
    sfs["HMER indel > 12"] = lambda x: x.indel & (x["x_hil"].apply(lambda y: y[0]) > 12)
    return sfs


class VariantSelectionFunctions(Enum):
    """Collecton of variant selection functions - all get DF as input and return boolean np.array"""

    @staticmethod
    def ALL(df: pd.DataFrame) -> np.ndarray:
        return np.ones(df.shape[0], dtype=bool)

    @staticmethod
    def HMER_INDEL(df: pd.DataFrame) -> np.ndarray:
        return np.array(df.hmer_indel_length > 0)

    @staticmethod
    def ALL_except_HMER_INDEL_greater_than_or_equal_5(df: pd.DataFrame) -> np.ndarray:
        return np.array(~((df.hmer_indel_length >= 5)))


def combine_multiallelic_spandel(df: pd.DataFrame, df_unsplit: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Combine predictions and scores for multiallelic variants and variants with spanning deletions. Since the
    genotyping model is trained to spit likelihoods of only three genotypes, we split the multiallelic VCF records
    into multiple rows with two alleles in each (see `multiallelics.split_multiallelic_variants` for the details).
    In this function we combine the multiple rows into a single one.

    Note that since the function receives the dataframe before the splitting and the split dataframe rows point
    to the original rows we only need to combine the PLs of the split rows.

    Parameters
    ----------
    df: pd.DataFrame
        Multiallelic spandel dataframe
    df_unsplit: pd.DataFrame
        Unsplit dataframe (the original read from the VCF)
    scores: np.ndarray
        Scores, in the order of the split dataframe

    Returns
    -------
    pd.DataFrame:
        df_unsplit with ml_lik column added representing the merged likelihoods
    """

    multiallelics = df[~pd.isnull(df["multiallelic_group"])].groupby("multiallelic_group")
    multiallelic_scores = scores[~pd.isnull(df["multiallelic_group"]), :]
    multiallelic_scores = pd.Series(
        [list(x) for x in multiallelic_scores], index=df[~pd.isnull(df["multiallelic_group"])].index
    )
    df_unsplit = merge_and_assign_pls(df_unsplit, multiallelics, multiallelic_scores)
    spandels = df[~pd.isnull(df["spanning_deletion"])].groupby(["chrom", "pos"])
    spandel_scores = scores[~pd.isnull(df["spanning_deletion"]), :]
    spandel_scores = pd.Series([list(x) for x in spandel_scores], index=df[~pd.isnull(df["spanning_deletion"])].index)
    df_unsplit = merge_and_assign_pls(df_unsplit, spandels, spandel_scores)
    return df_unsplit


def merge_and_assign_pls(
    original_df: pd.DataFrame, grouped_split_df: DataFrameGroupBy, split_scores: pd.Series
) -> pd.DataFrame:
    """Merges the PL scores of multiallelic and spanning deletion calls. The asprocess is like this:
    In the step of splitting the multiallelics, we have generated the rows as follows:
    Pick the "strongest" allele (e.g. A1),
    Combine the support for A1 and A2 into a single allele A
    Genotype (R,A): the result is (PL'(R,R), PL'(R,A), PL'(A,A))
    Genotype (A1,A2), where A1 is the reference, A2 is the "alt".
    In this function we are merging the split genotyping rows back as follows:
    Merge the support in the following way to generate PL(R,R),PL(R,A1),PL(A1,A1),PL(R,A2),PL(A1,A2),PL(A2,A2) tuple
    PL(R,R) = PL'(R,R)
    PL(R,A1) = PL'(R,A)
    PL(A1,A1) = PL'(A,A) * PL'(A1,A1)
    PL(A1,A2) = PL'(A,A) * PL(A1,A2)
    PL(A2,A2) = PL(A,A) * PL(A2,A2)

    Note that PL(R,A2) is zero here.

    Parameters
    ----------
    original_df: pd.DataFrame
        Original dataframe
    grouped_split_df: DataFrameGroupBy
        Dataframe of calls, where the multiallelics are split between the records,
        the dataframe is grouped by the multiallelics original locations
    split_scores: pd.Series
        Scores of the multiallelic variants

    Returns
    -------
    pd.DataFrame:
        The dataframe with multiallelics merged into a single record

    See also
    --------
    multiallelics.split_multiallelic_variants
    """
    result = []
    for g in grouped_split_df.groups:
        k = g
        rg = grouped_split_df.get_group(k)
        if rg.shape[0] == 1:
            pls = split_scores[grouped_split_df.groups[g][0]]
        else:
            orig_alleles = original_df.at[k, "alleles"]
            n_alleles = len(orig_alleles)
            n_pls = n_alleles * (n_alleles + 1) / 2
            pls = np.zeros(int(n_pls))
            idx1 = orig_alleles.index(rg.iloc[1]["alleles"][0])
            idx2 = orig_alleles.index(rg.iloc[1]["alleles"][1])
            tmp = split_scores[grouped_split_df.groups[g][0]][2] * np.array(split_scores[grouped_split_df.groups[g][1]])
            tmp = np.insert(tmp, 1, 0)
            merge_pls = np.concatenate((split_scores[grouped_split_df.groups[g][0]][:2], tmp))
            set_idx = [
                mu.get_pl_idx(x) for x in ((0, 0), (0, idx1), (idx1, idx1), (0, idx2), (idx1, idx2), (idx2, idx2))
            ]
            pls[set_idx] = merge_pls
        result.append(pls)

    set_dest = list(grouped_split_df.groups.keys())
    original_df.loc[set_dest, "ml_lik"] = pd.Series(result, index=original_df.loc[set_dest].index)
    return original_df
