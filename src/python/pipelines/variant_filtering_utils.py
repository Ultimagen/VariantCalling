from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn import impute
import sklearn_pandas
import pandas as pd
import numpy as np
import tqdm
import logging
from typing import Optional, Tuple, Callable, Union
from enum import Enum
import python.utils as utils
import sklearn
import matplotlib.pyplot as plt


FEATURES = ['sor', 'dp', 'qual', 'hmer_indel_nuc',
            'inside_hmer_run', 'close_to_hmer_run', 'hmer_indel_length','indel_length',
            'ad','af', 'fs','qd','mq','pl','gt',
            'gq','ps','ac','an',
            'baseqranksum','excesshet', 'mleac', 'mleaf', 'mqranksum', 'readposranksum','xc',
            'indel','left_motif','right_motif','alleles','cycleskip_status','gc_content']

logger = logging.getLogger(__name__)
class SingleModel:

    def __init__(self, threshold_dict: dict, is_greater_then: dict):
        self.threshold_dict = threshold_dict
        self.is_greater_then = is_greater_then

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_vec = np.ones(df.shape[0], dtype=np.bool)
        for v in self.threshold_dict:
            result_vec = result_vec & (
                (df[v] > self.threshold_dict[v]) == self.is_greater_then[v])
        return np.where(np.array(result_vec), "tp", 'fp')


class SingleRegressionModel:

    def __init__(self, threshold_dict: dict, is_greater_then: dict, score_translation: list):
        self.threshold_dict = threshold_dict
        self.is_greater_then = is_greater_then
        self.score = score_translation

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_vec = np.ones(df.shape[0], dtype=np.bool)
        results = []
        for v in self.threshold_dict:
            result_v = (np.array(df[v])[:, np.newaxis] > self.threshold_dict[v]
                        [np.newaxis, :]) == self.is_greater_then[v]
            results.append(result_v)
        result_vec = np.all(results, axis=0)
#        scores = self.score[np.argmax(result_vec == 0, axis=1)]
        scores = result_vec.mean(axis=1)
        return scores


class SingleTrivialClassifierModel:

    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pf = df['filter'].apply(lambda x: 'PASS' in x)
        return np.where(np.array(pf), "tp", "fp")


class SingleTrivialRegressorModel:

    def __init__(self):
        pass

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df.tree_score


class MaskedHierarchicalModel:

    def __init__(self, _name: str, _group_column: str, _models_dict: dict,
                 transformer: Optional[sklearn_pandas.DataFrameMapper] = None, tree_score_fpr=None):
        self.name = _name
        self.group_column = _group_column
        self.models = _models_dict
        self.transformer = transformer
        self.tree_score_fpr = tree_score_fpr

    def predict(self, df: pd.DataFrame,
                mask_column: Optional[str] = None) -> pd.Series:
        '''Makes prediction on the dataframe, optionally ignoring false-negative calls

        Parameters
        ----------
        df: pd.DataFrame
            Input dataframe
        mask_column: str or None (default)
            Column to look at to determine if the variant is false-negative

        Returns
        -------
        pd.Series
            Series of the size - number of rows in df, that contains tp/fp according to the model
        '''
        if mask_column is not None:
            mask = (df[mask_column] == 'fn')
        else:
            mask = np.zeros(df.shape[0], dtype=np.bool)

        apply_df = df[~mask]
        groups = set(df[self.group_column])
        gvecs = [df[self.group_column] == g for g in groups]
        result = pd.Series(['fn'] * df.shape[0], index=df.index)
        for i, g in enumerate(groups):
            result[(~mask) & (gvecs[i])] = self._predict_by_blocks(
                self.models[g], apply_df[apply_df[self.group_column] == g])
        return result

    def _predict_by_blocks(self, model, df):
        predictions = []
        for i in range(0, df.shape[0], 1000000):
            if self.transformer is not None:
                predictions.append(model.predict(
                    self.transformer.fit_transform(df.iloc[i:i + 1000000, :])))
            else:
                predictions.append(model.predict(df.iloc[i:i + 1000000, :]))
        return np.hstack(predictions)


def train_threshold_models(concordance: pd.DataFrame, interval_size: int, classify_column: str = 'classify', annots: list = [])\
        -> Tuple[MaskedHierarchicalModel, MaskedHierarchicalModel, pd.DataFrame]:
    '''Trains threshold classifier and regressor

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    interval_size: int
        number of bases in the interval
    classify_column: str
        Classification column
    annots: list
        Annotation interval names

    Returns
    -------
    tuple
        Tuple of classifier/regressor models
    '''

    train_selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(
        concordance, train_selection_functions, "group")
    concordance = add_testing_train_split_column(
        concordance, "group", "test_train_split", classify_column)
    transformer = feature_prepare(output_df=True, annots=annots)
    transformer.fit(concordance)
    groups = set(concordance["group"])
    classifier_models = {}
    regressor_models = {}
    fpr_values = {}
    for g in groups:
        classifier_models[g], regressor_models[g], fpr_values[g] = \
            train_threshold_model(concordance, concordance['test_train_split'],
                                  concordance['group'] == g, classify_column, transformer, interval_size, annots)

    return MaskedHierarchicalModel("Threshold classifier", "group", classifier_models, transformer=transformer), \
        MaskedHierarchicalModel("Threshold regressor", "group", regressor_models, transformer=transformer,
                                tree_score_fpr=fpr_values), \
        concordance


def train_threshold_model(concordance: pd.DataFrame, test_train_split: pd.Series,
                          selection: pd.Series, gtr_column: str,
                          transformer: sklearn_pandas.DataFrameMapper,
                          interval_size: int,
                          annots: list = []) -> tuple:
    '''Trains threshold regressor and classifier models

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
    '''

    quals = np.linspace(0, 500, 49)
    sors = np.linspace(0, 10, 49)

    pairs_qual_sor_threshold = [(quals[i], sors[j])
                                for i in range(len(quals)) for j in range(len(sors))]

    fns = np.array(concordance[gtr_column] == 'fn')
    train_data = concordance[selection & (~fns) & test_train_split][FEATURES + annots]

    train_data = transformer.transform(train_data)
    _validate_data(train_data.to_numpy())
    labels = concordance[selection & (~fns) & test_train_split][gtr_column]
    _validate_data(labels.to_numpy())
    enclabels = np.array(labels == 'tp')
    train_qual = train_data['qual']
    train_sor = train_data['sor']

    qq = (train_qual.to_numpy()[:, np.newaxis] > quals[np.newaxis, :])
    ss = (train_sor.to_numpy()[:, np.newaxis] < sors[np.newaxis, :])
    predictions_tp = (qq[..., np.newaxis] & ss[:, np.newaxis, :])
    tps = (predictions_tp & enclabels[:, np.newaxis, np.newaxis]).sum(axis=0)
    fns = ((~predictions_tp) & enclabels[
           :, np.newaxis, np.newaxis]).sum(axis=0)
    fps = (predictions_tp & (
        ~enclabels[:, np.newaxis, np.newaxis])).sum(axis=0)

    recalls = tps / (tps + fns)
    precisions = (tps + 1) / (tps + fps + 1)
    results_df = pd.DataFrame(data=np.vstack((recalls.flat, precisions.flat)).T,
                              index=pairs_qual_sor_threshold, columns=[('recall', 'var'), ('precision', 'var')])

    f1 = 2*results_df[('recall', 'var')] * results_df[('precision', 'var')] / \
                    (results_df[('recall', 'var')] + results_df[('precision', 'var')])
    results_df['f1'] = f1
    best = results_df['f1'].idxmax()
    classifier = SingleModel(dict(zip(['qual', 'sor'], best)), {
                             'sor': False, 'qual': True})
    rsi = get_r_s_i(results_df, 'var')[-1].copy()
    rsi.sort_values(('precision', 'var'), inplace=True)
    rsi['score'] = np.linspace(0, 1, rsi.shape[0])
    regression_model = SingleRegressionModel({'qual': np.array([x[0] for x in rsi.index]),
                                              'sor': np.array([x[1] for x in rsi.index])},
                                             {'sor': False, 'qual': True},
                                             np.array(rsi['score']))
    tree_scores = regression_model.predict(train_data)
    tree_scores_sorted, fpr_values = fpr_tree_score_mapping(
        tree_scores, labels, test_train_split[selection], interval_size)
    return classifier, regression_model, pd.concat([pd.Series(tree_scores_sorted), fpr_values], axis=1)


def get_r_s_i(results: pd.DataFrame, var_type: pd.DataFrame) -> tuple:
    '''Returns data for plotting ROC curve

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
    '''

    recall = results[('recall', var_type)]
    precision = results[('precision', var_type)]
    idx = utils.max_merits(np.array(recall), np.array(precision))
    results_plot = results.iloc[idx]
    return recall, precision, idx, results_plot


def get_all_precision_recalls(results: pd.DataFrame) -> dict:
    '''Return all precision-recall curves for thresholded models

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
    '''
    groups = set([x[1] for x in results.columns])
    result = {}
    for g in groups:
        tmp = get_r_s_i(results, g)[-1]
        result[g] = np.array(
            np.vstack((tmp[('recall', g)], tmp[('precision', g)]))).T
    return result


def calculate_threshold_model(results):
    all_groups = set([x[1] for x in results.columns])
    models = {}
    recalls_precisions = {}
    for g in all_groups:
        recall_series = results[('recall', g)]
        precision_series = results[('precision', g)]
        qual, sor = ((recall_series - 1)**2 +
                     (precision_series - 1)**2).idxmin()
        recalls_precisions[g] = recall_series[
            (qual, sor)], precision_series[(qual, sor)]
        models[g] = SingleModel({'qual': qual, 'sor': sor}, {
                                'qual': True, 'sor': False})
    result = MaskedHierarchicalModel("thresholding", 'group', models)
    return result, recalls_precisions


def tuple_break(x):
    '''Returns the first element in the tuple
    '''
    if type(x) == tuple:
        return x[0]
    return 0 if np.isnan(x) else x

def tuple_break_second(x):
    '''Returns the second element in the tuple
    '''
    if type(x) == tuple:
        return x[1]
    return 0 if np.isnan(x) else x


def motif_encode_left(x):
    '''Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    '''
    bases = {'A':1,
             'T':2,
             'G':3,
             'C':4,
             'N':5}
    x_list = list(x)
    x_list.reverse()
    num=0
    for c in x_list:
        num = 10 * num + bases.get(c,0)
    return num

def motif_encode_right(x):
    '''Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    '''
    bases = {'A':1,
             'T':2,
             'G':3,
             'C':4,
             'N':5}
    x_list = list(x)
    num=0
    for c in x_list:
        num = 10 * num + bases.get(c,0)
    return num

def allele_encode(x):
    '''Translate base into integer.
    In case we don't get a single base, we return zero
    '''
    bases = {'A':1,
             'T':2,
             'G':3,
             'C':4}
    return bases.get(x,0)

def gt_encode(x):
    '''Checks whether the variant is heterozygous(0) or homozygous(1)
    '''
    if x == (1,1):
        return 1
    return 0


def feature_prepare(output_df: bool = False, annots: list = []) -> sklearn_pandas.DataFrameMapper:
    '''Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    output_df: bool
        Should the transformer output dataframe (for threshold models) or numpy array (for trees)

    Returns
    -------
    tuple
        Mapper, list of features
    '''
    default_filler = impute.SimpleImputer(strategy='constant', fill_value=0)
    tuple_filter = sklearn_pandas.FunctionTransformer(tuple_break)
    tuple_filter_second = sklearn_pandas.FunctionTransformer(tuple_break_second)
    left_motif_filter = sklearn_pandas.FunctionTransformer(motif_encode_left)
    right_motif_filter = sklearn_pandas.FunctionTransformer(motif_encode_right)
    allele_filter = sklearn_pandas.FunctionTransformer(allele_encode)
    gt_filter = sklearn_pandas.FunctionTransformer(gt_encode)

    transform_list = [(['sor'], default_filler),
                      (['dp'], default_filler),
                      ('qual', None),
                      ('hmer_indel_nuc', preprocessing.LabelEncoder()),
                      ('inside_hmer_run', None),
                      ('close_to_hmer_run', None),
                      (['hmer_indel_length'], default_filler),
                      (['indel_length'],default_filler),
                      ('ad', [tuple_filter]),
                      ('af', [tuple_filter]),
                      (['fs'], default_filler),
                      (['qd'], default_filler),
                      (['mq'], default_filler),
                      ('pl', [tuple_filter]),
                      ('gt', [gt_filter]),
                      (['gq'], default_filler),
                      (['ps'], default_filler),
                      ('ac', [tuple_filter]),
                      (['an'], default_filler),
                      (['baseqranksum'], default_filler),
                      (['excesshet'], default_filler),
                      ('mleac', [tuple_filter]),
                      ('mleaf', [tuple_filter]),
                      (['mqranksum'], default_filler),
                      (['readposranksum'], default_filler),
                      (['xc'], default_filler),
                      ('indel', None),
                      ('left_motif', [left_motif_filter]),
                      ('right_motif', [right_motif_filter]),
                      ('cycleskip_status', preprocessing.LabelEncoder()),
                      ('alleles', [tuple_filter, allele_filter]),
                      ('alleles', [tuple_filter_second, allele_filter]),
                      (['gc_content'], default_filler)
                      ]
    for annot in annots:
        transform_list.append((annot, None))

    transformer = sklearn_pandas.DataFrameMapper(
        transform_list, df_out=output_df)
    return transformer


def train_model(concordance: pd.DataFrame, test_train_split: np.ndarray,
                selection: pd.Series, gtr_column: str,
                transformer: sklearn_pandas.DataFrameMapper,
                interval_size: int,
                classify_model,
                regression_model,
                annots: list = [],
                exome_weight: int = 1,
                exome_weight_annotation: str = None) -> Tuple[DecisionTreeClassifier, DecisionTreeRegressor, pd.DataFrame]:
    '''Trains model on a subset of dataframe that is already dividied into a testing and training set

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

    Returns
    -------
    tuple:
        Trained classifier model, trained regressor model
    '''
    fns = np.array(concordance[gtr_column] == 'fn')
    train_data = concordance[test_train_split & selection & (~fns)][FEATURES + annots]

    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    train_data = transformer.transform(train_data)

    _validate_data(train_data)
    _validate_data(labels.to_numpy())

    model = classify_model
    model1 = regression_model
    enclabels = preprocessing.LabelEncoder().fit_transform(labels)

    if exome_weight != 1 and (exome_weight_annotation is not None) and\
            isinstance(model, RandomForestClassifier):
        sample_weight = concordance[test_train_split & selection & (~fns)][FEATURES + annots][exome_weight_annotation]
        sample_weight = sample_weight.apply(lambda x: exome_weight if x else 1)

        model.fit(train_data, labels, sample_weight=sample_weight)
        model1.fit(train_data, enclabels, sample_weight=sample_weight)
    else:
        model.fit(train_data, labels)
        model1.fit(train_data, enclabels)

    tree_scores = model1.predict(train_data)
    if gtr_column == 'classify':  ## there is gt
        tree_scores_sorted, fpr_values = fpr_tree_score_mapping(
            tree_scores, labels, test_train_split, interval_size)
        return model, model1, pd.concat([pd.Series(tree_scores_sorted), fpr_values], axis=1,)
    else:
        return model, model1, None

def train_model_NN(concordance: pd.DataFrame, test_train_split: np.ndarray,
                selection: pd.Series, gtr_column: str,
                transformer: sklearn_pandas.DataFrameMapper,
                interval_size: int, annots: list = [],
                exome_weight: int = 1,
                exome_weight_annotation: str = None) -> Tuple[DecisionTreeClassifier, DecisionTreeRegressor, pd.DataFrame]:
    '''Trains model on a subset of dataframe that is already divided into a testing and training set

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
    Returns
    -------
    tuple:
        Trained classifier model, trained regressor model
    '''

    model = MLPClassifier(solver='sgd', alpha=1e-5,
        hidden_layer_sizes=(3,), random_state=1)

    model1 = MLPRegressor(solver='sgd', alpha=1e-5,
        hidden_layer_sizes=(3,), random_state=1)
    return train_model(concordance, test_train_split,
                   selection, gtr_column, transformer, interval_size, model, model1, annots=annots,
                       exome_weight=exome_weight,
                       exome_weight_annotation=exome_weight_annotation)


def train_model_RF(concordance: pd.DataFrame, test_train_split: np.ndarray,
                   selection: pd.Series, gtr_column: str,
                   transformer: sklearn_pandas.DataFrameMapper,
                   interval_size: int, annots: list = [],
                   exome_weight: int = 1,
                   exome_weight_annotation: str = None) -> Tuple[DecisionTreeClassifier, DecisionTreeRegressor, pd.DataFrame]:
    '''Trains model on a subset of dataframe that is already divided into a testing and training set

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
    Returns
    -------
    tuple:
        Trained classifier model, trained regressor model
    '''
    model = RandomForestClassifier(n_estimators=40,max_depth=8)

    model1 = RandomForestRegressor(n_estimators=40,max_depth=8)
    return train_model(concordance, test_train_split,
                   selection, gtr_column, transformer, interval_size, model, model1, annots=annots,
                       exome_weight=exome_weight,
                       exome_weight_annotation=exome_weight_annotation)

def train_model_DT(concordance: pd.DataFrame, test_train_split: np.ndarray,
                   selection: pd.Series, gtr_column: str,
                   transformer: sklearn_pandas.DataFrameMapper,
                   interval_size: int, annots: list = [],
                   exome_weight: int = 1,
                   exome_weight_annotation: str = None) -> Tuple[DecisionTreeClassifier, DecisionTreeRegressor, pd.DataFrame]:
    '''Trains model on a subset of dataframe that is already divided into a testing and training set

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
    Returns
    -------
    tuple:
        Trained classifier model, trained regressor model
    '''
    model = DecisionTreeClassifier(max_depth=5)

    model1 = DecisionTreeRegressor(max_depth=5)
    return train_model(concordance, test_train_split,
                   selection, gtr_column, transformer, interval_size, model, model1, annots=annots,
                       exome_weight=exome_weight,
                       exome_weight_annotation=exome_weight_annotation)


def _validate_data(data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
    '''Validates that the data does not contain nulls'''

    if type(data) == np.ndarray:
        test_data = data
    else:
        test_data = data.to_numpy()
    try:
        if len(test_data.shape) == 1 or test_data.shape[1] <= 1:
            assert pd.isnull(test_data).sum() == 0, "data vector contains null"
        else:
            for c in range(test_data.shape[1]):
                assert pd.isnull(test_data[:, c]).sum() == 0, f"Data matrix contains null in column {c}"
    except AssertionError as af:
        logger.error(str(af))
        raise af


def fpr_tree_score_mapping(tree_scores: np.ndarray, labels: pd.Series, test_train_split: pd.Series, interval_size: int) -> pd.Series:
    '''Clclulate False Positive Rate for each variant
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
        '''
    # in case we do not run frp - interval_size is None
    if interval_size is None:
        return np.zeros(len(tree_scores)), pd.Series(np.zeros(len(tree_scores)))
    train_part = sum(test_train_split)/len(test_train_split)
    tree_scores_sorted_inds = np.argsort(tree_scores)
    cur_fpr = 0
    fpr = []
    for cur_ind in tree_scores_sorted_inds[::-1]:
        if labels[cur_ind] == 'fp':
            cur_fpr = cur_fpr+1
        fpr.append((cur_fpr/train_part) / interval_size)
    return tree_scores[tree_scores_sorted_inds], pd.Series(fpr[::-1]) * 10**6


def get_basic_selection_functions():
    'Selection between SNPs and INDELs'
    sfs = []
    names = []
    sfs.append(lambda x: np.logical_not(x.indel))
    names.append("snp")
    sfs.append(lambda x: (x.indel))
    names.append("indel")

    return dict(zip(names, sfs))


def get_training_selection_functions():
    '''
    '''
    sfs = []
    names = []
    sfs.append(lambda x: np.logical_not(x.indel))
    names.append("snp")
    sfs.append(lambda x: (x.indel & (x.hmer_indel_length == 0)))
    names.append("non-h-indel")
    sfs.append(lambda x: (x.indel & (x.hmer_indel_length > 0)))
    names.append("h-indel")
    return dict(zip(names, sfs))


def find_thresholds(concordance: pd.DataFrame, classify_column: str = 'classify', sf_generator: Callable = get_training_selection_functions) -> pd.DataFrame:
    quals = np.linspace(0, 2000, 30)
    sors = np.linspace(0, 20, 80)
    results = []
    pairs = []
    selection_functions = sf_generator()
    concordance = add_grouping_column(
        concordance, selection_functions, "group")
    for q in tqdm.tqdm_notebook(quals):
        for s in sors:
            pairs.append((q, s))
            tmp = (concordance[((concordance['qual'] > q) & (concordance['sor'] < s)) |
                               (concordance[classify_column] == 'fn')][[classify_column, 'group']]).copy()
            tmp1 = (concordance[((concordance['qual'] < q) | (concordance['sor'] > s)) &
                                (concordance[classify_column] == 'tp')][[classify_column, 'group']]).copy()
            tmp1[classify_column] = 'fn'
            tmp2 = pd.concat((tmp, tmp1))
            results.append(tmp2.groupby([classify_column, 'group']).size())
    results_df = pd.concat(results, axis=1)
    results_df = results_df.T
    results_df.columns = results_df.columns.to_flat_index()

    for group in set(concordance['group']):
        results_df[('recall', group)] = results_df.get(('tp', group), 0) / \
            (results_df.get(('tp', group), 0) +
             results_df.get(('fn', group), 0) + 1)
        results_df[('precision', group)] = results_df.get(('tp', group), 0) / \
            (results_df.get(('tp', group), 0) +
             results_df.get(('fp', group), 0) + 1)
        results_df.index = pairs
    return results_df


def add_grouping_column(df: pd.DataFrame, selection_functions: dict, column_name: str) -> pd.DataFrame:
    '''Add a column for grouping according to the values of selection functions

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
    '''
    df[column_name] = None
    for k in selection_functions:
        df.loc[selection_functions[k](df), column_name] = k
    return df


def tree_score_to_fpr(df: pd.DataFrame, prediction_score: pd.Series, tree_score_fpr: pd.DataFrame) -> pd.DataFrame:
    '''Deduce frp value from the tree_score and the tree score fpr mapping

        Parameters
        ----------
        df: pd.DataFrame
            concordance dataframe
        prediction_score: pd.Series
        tree_score_fpr: dict -> pd.DataFrame
            dictionary of group -> df were the df is
            2 columns of tree score and its corresponding fpr in increasing order of tree_score
            and the group key is snp, h-indel, non-h-indel

        Returns
        -------
        pd.DataFrame
            df with column_name added to it that is filled with the fpr value according
            to the tree score fpr mapping
        '''

    fpr_values = pd.Series(np.zeros(len(prediction_score)))
    fpr_values.index = prediction_score.index
    for group in df['group'].unique():
        select = df['group'] == group
        tree_score_fpr_group = tree_score_fpr[group]
        if tree_score_fpr_group is not None:
            # it is None in case we didn't run training on gt, but on dbsnp and blacklist
            fpr_values.loc[select] = np.interp(
                prediction_score.loc[select], tree_score_fpr_group.iloc[:, 0], tree_score_fpr_group.iloc[:, 1])
    return fpr_values


def get_testing_selection_functions() -> dict:
    sfs = []
    sfs.append(('SNP', lambda x: np.logical_not(x.indel)))
    sfs.append(("Non-hmer INDEL", lambda x: x.indel &
                (x.hmer_indel_length == 0)))
    sfs.append(("HMER indel <= 4", (lambda x: x.indel & (x.hmer_indel_length > 0) &
                                                        (x.hmer_indel_length < 5))))
    sfs.append(("HMER indel (4,8)", lambda x: x.indel & (x.hmer_indel_length >= 5) &
                (x.hmer_indel_length < 8)))
    sfs.append(("HMER indel [8,10]", lambda x: x.indel & (x.hmer_indel_length >= 8) &
                (x.hmer_indel_length <= 10)))

    sfs.append(("HMER indel 11,12", lambda x: x.indel & (x.hmer_indel_length >= 11) &
                (x.hmer_indel_length <= 12)))

    sfs.append(("HMER indel > 12", lambda x: x.indel &
                (x.hmer_indel_length > 12)))

    return dict(sfs)


def add_testing_train_split_column(concordance: pd.DataFrame,
                                   training_groups_column: str, test_train_split_column: str,
                                   gtr_column: str,
                                   min_test_set: int = 50, max_train_set: int = 200000,
                                   test_set_fraction: float = .5) -> pd.DataFrame:
    '''Adds a column that divides each training group into a train/test set. Supports
    requirements for the minimal testing set size, maximal training test size and the fraction of test

    Parameters
    ----------
    concordance: pd.DataFrame
        Input data frame
    testing_groups_column: str
        Name of the grouping column
    test_train_split_column: str
        Name of the splitting column
    gtr_column: str
        Name of the column that contains ground truth (will exclude fns)
    min_test_set: int
        Default - 2000
    max_train_set: int
        Default - 200000
    test_set_fraction: float
        Default - 0.5

    Returns
    -------
    pd.DataFrame
        Dataframe with 0/1 in test_train_split_column, with 1 for train
    '''
    groups = set(concordance[training_groups_column])

    test_train_split_vector = np.zeros(concordance.shape[0], dtype=np.bool)
    for g in groups:
        group_vector = (concordance[training_groups_column] == g)
        locations = group_vector.to_numpy().nonzero()[0]
        assert(group_vector.sum() >=
               min_test_set), "Group size too small for training"
        train_set_size = int(min(group_vector.sum() - min_test_set,
                                 max_train_set,
                                 group_vector.sum() * (1 - test_set_fraction)))
        test_set_size = group_vector.sum() - train_set_size
        assert(test_set_size >= min_test_set), \
            f"Test set size too small -> test:{test_set_size}, train:{train_set_size}"
        assert(train_set_size <= max_train_set), \
            f"Train set size too big -> test:{test_set_size}, train:{train_set_size}"
        train_set = locations[np.random.choice(np.arange(group_vector.sum(), dtype=np.int),
                                               train_set_size, replace=False)]
        test_train_split_vector[train_set] = True

    concordance[test_train_split_column] = test_train_split_vector
    return concordance

def train_model_wrapper(concordance: pd.DataFrame, classify_column: str, interval_size: int, train_function,
                        model_name: str,
                        annots: list = [],
                        exome_weight: int = 1,
                        exome_weight_annotation: str = None,
                        use_train_test_split: bool = True) -> tuple:
    '''Train a decision tree model on the dataframe

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


    Returns
    -------
    (MaskedHierarchicalModel, pd.DataFrame
        Models for each group, DataFrame with group for a hierarchy group and test_train_split columns

    '''
    logger.info("Train model " + model_name)
    train_selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(
        concordance, train_selection_functions, "group")
    if use_train_test_split:
        concordance = add_testing_train_split_column(
            concordance, "group", "test_train_split", classify_column)
    else:
        concordance['test_train_split'] = True
    transformer = feature_prepare(annots=annots)
    transformer.fit(concordance)
    groups = set(concordance["group"])
    classifier_models:dict = {}
    regressor_models:dict = {}
    fpr_values:dict = {}
    for g in groups:
        classifier_models[g], regressor_models[g], fpr_values[g] = \
            train_function(concordance, concordance['test_train_split'],
                        concordance['group'] == g, classify_column, transformer, interval_size, annots,
                           exome_weight=exome_weight,
                           exome_weight_annotation=exome_weight_annotation)

    return MaskedHierarchicalModel(model_name + " classifier", "group", classifier_models, transformer=transformer), \
        MaskedHierarchicalModel(model_name + " regressor", "group", regressor_models, transformer=transformer,
                                tree_score_fpr=fpr_values), \
        concordance


def test_decision_tree_model(concordance: pd.DataFrame, model: MaskedHierarchicalModel, classify_column: str) -> dict:
    '''Calculate precision/recall for the decision tree classifier

    Parameters
    ----------
    concordance: pd.DataFrame
        Input dataframe
    model: MaskedHierarchicalModel
        Model
    classify_column: str
        Ground truth labels

    Returns
    -------
    dict:
        Tuple dictionary - recall/precision for each category
    '''
    concordance = add_grouping_column(
        concordance, get_testing_selection_functions(), "group_testing")
    predictions = model.predict(concordance, classify_column)

    groups = set(concordance['group_testing'])
    recalls_precisions = {}
    for g in groups:
        select = (concordance["group_testing"] == g) & \
            (~concordance["test_train_split"])

        group_ground_truth = concordance.loc[select, classify_column]
        group_predictions = predictions[select]
        group_ground_truth[group_ground_truth == 'fn'] = 'tp'

        recall = metrics.recall_score(
            group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        precision = metrics.precision_score(
            group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        f1 = metrics.f1_score(
            group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        recalls_precisions[g] = (recall, precision,f1)

    return recalls_precisions


def get_decision_tree_precision_recall_curve(concordance: pd.DataFrame,
                                             model: MaskedHierarchicalModel,
                                             classify_column: str) -> dict:
    '''Calculate precision/recall curve for the decision tree regressor

    Parameters
    ----------
    concordance: pd.DataFrame
        Input dataframe
    model: MaskedHierarchicalModel
        Model
    classify_column: str
        Ground truth labels

    Returns
    -------
    dict:
        Tuple dictionary - recall/precision for each category
    '''

    concordance = add_grouping_column(
        concordance, get_testing_selection_functions(), "group_testing")
    predictions = model.predict(concordance, classify_column)
    groups = set(concordance['group_testing'])
    recalls_precisions = {}

    for g in groups:
        select = (concordance["group_testing"] == g) & \
                 (~concordance["test_train_split"])

        group_ground_truth = concordance.loc[select, classify_column]
        group_predictions = predictions[select] ## as type object
        group_predictions[group_ground_truth == 'fn'] = -1
        # this is a change to calculate recall correctly
        group_ground_truth[group_ground_truth == 'fn'] = 'tp'

        curve = utils.precision_recall_curve(np.array(group_ground_truth), np.array(
            group_predictions), pos_label="tp", fn_score=-1)
        # curve = metrics.precision_recall_curve(np.array(group_ground_truth), np.array(
        #    group_predictions), pos_label="tp")



        precision, recall, f1, preditions = curve

        recalls_precisions[g] = np.vstack((recall, precision, f1, preditions)).T

    return recalls_precisions


def calculate_unfiltered_model(concordance: pd.DataFrame, classify_column: str) -> tuple:
    '''Calculates precision and recall on the unfiltered data

    Parameters
    ----------
    concordance: pd.DataFrame
        Comparison dataframe
    classify_column: str
        Classification column

    Returns
    -------
    tuple:
        MaskedHierarchyclaModel, dict, model and dictionary of recalls_precisions
    '''

    selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(
        concordance, selection_functions, "group")
    all_groups = set(concordance['group'])
    models = {}
    for g in all_groups:
        models[g] = SingleModel({}, {})
    result = MaskedHierarchicalModel("unfiltered", 'group', models)
    concordance['test_train_split'] = np.zeros(
        concordance.shape[0], dtype=np.bool)
    recalls_precisions = test_decision_tree_model(
        concordance, result, classify_column)
    return result, recalls_precisions


class Blacklist(object):
    '''Class that stores the blacklist.

    Attributes
    ----------
    blacklist: set
        The blacklist of positions
    annotation: str
        Name of the blacklist
    selection_fcn: Callable
        The function that selects the relevant calls from the variant dataframe

    Parameters
    ---------
    blacklist: set
    annotation: str
    selection_fcn: Callable
    '''

    def __init__(self, blacklist: set, annotation: str, selection_fcn: Callable, description: str):
        self.blacklist = blacklist
        self.annotation = annotation
        self.selection_fcn = selection_fcn
        self.description = description

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Applies the blacklist on the dataframe

        Parameters
        ----------
        df : pd.DataFrame
            Input concordance dataframe

        Returns
        -------
        pd.Series
            Series with string annotation for the blacklist 
        """

        select = self.selection_fcn(df)
        idx = set(df[select].index)
        common_with_blacklist = idx & self.blacklist
        result = pd.Series("PASS", index=df.index, dtype=str)
        result.loc[common_with_blacklist] = self.annotation
        return result

    def __str__(self):
        return f"{self.annotation}: {self.description} with {len(self.blacklist)} elements"


def merge_blacklists(blacklists: list) -> pd.Series:
    """Combines blacklist annotations from multiple blacklists. Note that the merge
    does not make annotations unique and does not remove PASS from failed annotations

    Parameters
    ----------
    blacklists : list
        list of annotations from blacklist.apply

    Returns
    -------
    pd.Series
        Combined annotations
    """
    if len(blacklists) == 0:
        return None
    elif len(blacklists) == 1:
        return blacklists[0]

    concat = blacklists[0].str.cat(blacklists[1:], sep=";", na_rep="PASS")

    return concat


def blacklist_cg_insertions(df: pd.DataFrame) -> pd.Series:
    """Removes CG insertions from calls

    Parameters
    ----------
    df: pd.DataFrame
        calls concordance

    Returns
    -------
    pd.Series
    """
    ggc_filter = df['alleles'].apply(lambda x: 'GGC' in x or 'CCG' in x)
    blank = pd.Series("PASS", dtype=str, index=df.index)
    blank = blank.where(~ggc_filter, "CG_NON_HMER_INDEL")
    return blank


class VariantSelectionFunctions (Enum):
    """Collecton of variant selection functions - all get DF as input and return boolean np.array"""
    def ALL(
        df: pd.DataFrame) -> np.ndarray: return np.ones(df.shape[0], dtype=np.bool)

    def HMER_INDEL(
        df: pd.DataFrame) -> np.ndarray: return np.array(df.hmer_indel_length > 0)
    
    def ALL_except_HMER_INDEL_greater_than_or_equal_5(
        df: pd.DataFrame) -> np.ndarray: return np.array(~ ((df.hmer_indel_length >= 5)))