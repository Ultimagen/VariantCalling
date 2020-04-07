from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn import impute
import sklearn_pandas
import pandas as pd
import numpy as np
import tqdm
from typing import Optional, Tuple, Callable
import python.utils as utils

FEATURES = ['sor', 'dp', 'qual', 'hmer_indel_nuc',
            'inside_hmer_run', 'close_to_hmer_run', 'hmer_indel_length']


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
                 transformer: Optional[sklearn_pandas.DataFrameMapper]=None):
        self.name = _name
        self.group_column = _group_column
        self.models = _models_dict
        self.transformer = transformer

    def predict(self, df: pd.DataFrame,
                mask_column: Optional[str]=None) -> pd.Series:
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
            result[(~mask) & (gvecs[i])] = self._predict_by_blocks(self.models[g], apply_df[apply_df[self.group_column] == g])
        return result

    def _predict_by_blocks(self, model, df): 
        predictions = []
        for i in range(0,df.shape[0], 1000000):
            if self.transformer is not None : 
                predictions.append(model.predict(self.transformer.fit_transform(df.iloc[i:i+1000000,:])))
            else:
                predictions = model.predict(df.iloc[i:i+1000000,:])
        return np.hstack(predictions)

def train_threshold_models(concordance: pd.DataFrame, classify_column: str = 'classify')\
        -> Tuple[MaskedHierarchicalModel, MaskedHierarchicalModel, pd.DataFrame]:
    '''Trains threshold classifier and regressor

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    classify_column: str
        Classification column

    Returns
    -------
    tuple
        Tuple of classifier/regressor models
    '''

    train_selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(concordance, train_selection_functions, "group")
    concordance = add_testing_train_split_column(concordance, "group", "test_train_split", classify_column)
    transformer = feature_prepare(output_df=True)
    transformer.fit(concordance)
    groups = set(concordance["group"])
    classifier_models = {}
    regressor_models = {}
    for g in groups:
        classifier_models[g], regressor_models[g] = \
            train_threshold_model(concordance, concordance['test_train_split'],
                                  concordance['group'] == g, classify_column, transformer)

    return MaskedHierarchicalModel("Threshold classifier", "group", classifier_models, transformer=transformer), \
        MaskedHierarchicalModel("Threshold regressor", "group", regressor_models, transformer=transformer), \
        concordance


def train_threshold_model(concordance: pd.DataFrame, test_train_split: pd.Series,
                          selection: pd.Series, gtr_column: str,
                          transformer: sklearn_pandas.DataFrameMapper) -> tuple:
    '''Trains threshold regressor and classifier models

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    test_train_split: np.ndarray
        Test train split column
    selection : pd.Series
        Boolean - rows of concordance that belong to the group being trained
    gtr_column: str
        Ground truth column
    transformer: sklearn_pandas.DataFrameMapper
        Feature mapper
    '''

    quals = np.linspace(0, 2000, 30)
    sors = np.linspace(0, 20, 80)

    pairs_qual_sor_threshold = [(quals[i], sors[j]) for i in range(len(quals)) for j in range(len(sors))]

    fns = np.array(concordance[gtr_column] == 'fn')
    train_data = concordance[test_train_split & selection & (~fns)][FEATURES]

    train_data = transformer.transform(train_data)
    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    enclabels = np.array(labels == 'tp')
    train_qual = train_data['qual']
    train_sor = train_data['sor']

    qq = (train_qual[:, np.newaxis] > quals[np.newaxis, :])
    ss = (train_sor[:, np.newaxis] < sors[np.newaxis, :])
    predictions_tp = (qq[..., np.newaxis] & ss[:, np.newaxis, :])
    tps = (predictions_tp & enclabels[:, np.newaxis, np.newaxis]).sum(axis=0)
    fns = ((~predictions_tp) & enclabels[:, np.newaxis, np.newaxis]).sum(axis=0)
    fps = (predictions_tp & (~enclabels[:, np.newaxis, np.newaxis])).sum(axis=0)

    recalls = tps / (tps + fns)
    precisions = (tps + 1) / (tps + fps + 1)
    results_df = pd.DataFrame(data=np.vstack((recalls.flat, precisions.flat)).T,
                              index=pairs_qual_sor_threshold, columns=[('recall', 'var'), ('precision', 'var')])

    dist = (results_df[('recall', 'var')] - 1)**2 + (results_df[('precision', 'var')] - 1)**2
    results_df['dist'] = dist
    best = results_df['dist'].idxmin()
    classifier = SingleModel(dict(zip(['qual', 'sor'], best)), {'sor': False, 'qual': True})
    rsi = get_r_s_i(results_df, 'var')[-1].copy()
    rsi.sort_values(('precision', 'var'), inplace=True)
    rsi['score'] = np.linspace(0, 1, rsi.shape[0])
    regression_model = SingleRegressionModel({'qual': np.array([x[0] for x in rsi.index]),
                                              'sor': np.array([x[1] for x in rsi.index])},
                                             {'sor': False, 'qual': True},
                                             np.array(rsi['score']))
    return classifier, regression_model



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
        result[g] = np.array(np.vstack((tmp[('recall', g)], tmp[('precision', g)]))).T
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
        recalls_precisions[g] = recall_series[(qual, sor)], precision_series[(qual, sor)]
        models[g] = SingleModel({'qual': qual, 'sor': sor}, {
                                'qual': True, 'sor': False})
    result = MaskedHierarchicalModel("thresholding", 'group', models)
    return result, recalls_precisions


def feature_prepare(output_df: bool = False) -> sklearn_pandas.DataFrameMapper:
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
    transform_list = [(['sor'], default_filler),
                      (['dp'], default_filler),
                      ('qual', None),
                      ('inside_hmer_run', None),
                      ('close_to_hmer_run', None),
                      ('hmer_indel_nuc', preprocessing.LabelEncoder()),
                      (['hmer_indel_length'], default_filler)]
    transformer = sklearn_pandas.DataFrameMapper(transform_list, df_out=output_df)
    return transformer


def train_model(concordance: pd.DataFrame, test_train_split: np.ndarray,
                selection: pd.Series, gtr_column: str,
                transformer: sklearn_pandas.DataFrameMapper) -> Tuple[DecisionTreeClassifier, DecisionTreeRegressor]:
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
    Returns
    -------
    tuple:
        Trained classifier model, trained regressor model
    '''
    fns = np.array(concordance[gtr_column] == 'fn')
    train_data = concordance[test_train_split & selection & (~fns)][FEATURES]
    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    train_data = transformer.transform(train_data)

    model = DecisionTreeClassifier(max_depth=7)
    model.fit(train_data, labels)

    model1 = DecisionTreeRegressor(max_depth=7)
    enclabels = preprocessing.LabelEncoder().fit_transform(labels)
    model1.fit(train_data, enclabels)
    return model, model1


def get_basic_selection_functions() : 
    'Selection between SNPs and INDELs'
    sfs = []
    names = []
    sfs.append(lambda x: (~x.indel))
    names.append("snp")
    sfs.append(lambda x: (x.indel))
    names.append("indel")

    return dict(zip(names, sfs))


def get_training_selection_functions():
    '''
    '''
    sfs = []
    names = []
    sfs.append(lambda x: (~x.indel))
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
    concordance = add_grouping_column(concordance, selection_functions, "group")
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
            (results_df.get(('tp', group), 0) + results_df.get(('fn', group), 0) + 1)
        results_df[('precision', group)] = results_df.get(('tp', group), 0) / \
            (results_df.get(('tp', group), 0) + results_df.get(('fp', group), 0) + 1)
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


def get_testing_selection_functions() -> dict:
    sfs = []
    sfs.append(('SNP', lambda x: ~x.indel))
    #sfs.append(("INDEL", lambda x: x.indel))
    sfs.append(("Non-hmer INDEL", lambda x: x.indel & (x.hmer_indel_length == 0)))
    sfs.append(("HMER indel <= 4", (lambda x: x.indel & (x.hmer_indel_length > 0) &
                                                        (x.hmer_indel_length < 5))))
    sfs.append(("HMER indel > 4, < 12", lambda x: x.indel & (x.hmer_indel_length >= 5) &
                (x.hmer_indel_length < 12)))
    sfs.append(("HMER indel > 12", lambda x: x.indel & (x.hmer_indel_length >= 12)))
    #sfs.append(("HMER indel", lambda x: x.indel & (x.hmer_indel_length > 0)))

    return dict(sfs)


def add_testing_train_split_column(concordance: pd.DataFrame,
                                   training_groups_column: str, test_train_split_column: str,
                                   gtr_column: str,
                                   min_test_set: int = 2000, max_train_set: int = 200000,
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
        assert(group_vector.sum() > min_test_set), "Group size too small for training"
        train_set_size = int(min(group_vector.sum() - min_test_set,
                                 max_train_set,
                                 group_vector.sum() * (1 - test_set_fraction)))
        test_set_size = group_vector.sum() - train_set_size
        assert(test_set_size > min_test_set), \
            f"Test set size too small -> test:{test_set_size}, train:{train_set_size}"
        assert(train_set_size < max_train_set), \
            f"Train set size too big -> test:{test_set_size}, train:{train_set_size}"
        train_set = locations[np.random.choice(np.arange(group_vector.sum(), dtype=np.int),
                                               train_set_size, replace=False)]
        test_train_split_vector[train_set] = True

    concordance[test_train_split_column] = test_train_split_vector
    return concordance


def train_decision_tree_model(concordance: pd.DataFrame, classify_column: str) -> tuple:
    '''Train a decision tree model on the dataframe

    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe
    classify_column: str
        Ground truth labels
    Returns
    -------
    (MaskedHierarchicalModel, pd.DataFrame
        Models for each group, DataFrame with group for a hierarchy group and test_train_split columns

    '''

    train_selection_functions = get_training_selection_functions()
    concordance = add_grouping_column(concordance, train_selection_functions, "group")
    concordance = add_testing_train_split_column(concordance, "group", "test_train_split", classify_column)
    transformer = feature_prepare()
    transformer.fit(concordance)
    groups = set(concordance["group"])
    classifier_models = {}
    regressor_models = {}
    for g in groups:
        classifier_models[g], regressor_models[g] = \
            train_model(concordance, concordance['test_train_split'],
                        concordance['group'] == g, classify_column, transformer)

    return MaskedHierarchicalModel("Decision tree classifier", "group", classifier_models, transformer=transformer), \
        MaskedHierarchicalModel("Decision tree regressor", "group", regressor_models, transformer=transformer), \
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
    concordance = add_grouping_column(concordance, get_testing_selection_functions(), "group_testing")
    predictions = model.predict(concordance, classify_column)

    groups = set(concordance['group_testing'])
    recalls_precisions = {}
    for g in groups:
        select = (concordance["group_testing"] == g) & \
                 (~concordance["test_train_split"])
        group_ground_truth = concordance.loc[select, classify_column]
        group_predictions = predictions[select]
        group_ground_truth[group_ground_truth == 'fn'] = 'tp'

        recall = metrics.recall_score(group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        precision = metrics.precision_score(group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        recalls_precisions[g] = (recall, precision)

    return recalls_precisions


def get_decision_tree_precision_recall_curve(concordance: pd.DataFrame, model: MaskedHierarchicalModel,
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
    concordance = add_grouping_column(concordance, get_testing_selection_functions(), "group_testing")
    predictions = model.predict(concordance, classify_column)
    groups = set(concordance['group_testing'])
    recalls_precisions = {}

    for g in groups:
        select = (concordance["group_testing"] == g) & \
                 (~concordance["test_train_split"])
        group_ground_truth = concordance.loc[select, classify_column]
        group_predictions = predictions[select]
        group_predictions[group_ground_truth == 'fn'] = -1
        group_ground_truth[group_ground_truth == 'fn'] = 'tp'  # this is a change to calculate recall correctly

        curve = utils.precision_recall_curve(np.array(group_ground_truth), np.array(
            group_predictions), pos_label="tp", fn_score=-1)
        # curve = metrics.precision_recall_curve(np.array(group_ground_truth), np.array(
        #    group_predictions), pos_label="tp")

        precision, recall = curve

        recalls_precisions[g] = np.vstack((recall, precision)).T

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
    concordance = add_grouping_column(concordance, selection_functions, "group")
    all_groups = set(concordance['group'])
    models = {}
    for g in all_groups:
        models[g] = SingleModel({}, {})
    result = MaskedHierarchicalModel("unfiltered", 'group', models)
    concordance['test_train_split'] = np.zeros(concordance.shape[0], dtype=np.bool)
    recalls_precisions = test_decision_tree_model(concordance, result, classify_column)
    return result, recalls_precisions
