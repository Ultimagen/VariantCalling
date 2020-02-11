from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # type: ignore
from sklearn import preprocessing  # type: ignore
from sklearn import metrics
from sklearn import impute
import sklearn_pandas
import pandas as pd
import numpy as np
import tqdm
from typing import Callable, Optional
import python.utils as utils

FEATURES = ['sor', 'dp', 'qual', 'hmer_indel_nuc',
            'inside_hmer_run', 'close_to_hmer_run']


class SingleModel:
    def __init__(self, threshold_dict, is_greater_then):
        self.threshold_dict = threshold_dict
        self.is_greater_then = is_greater_then

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_vec = np.ones(df.shape[0], dtype=np.bool)
        for v in self.threshold_dict:
            result_vec = result_vec & (
                (df[v] > self.threshold_dict[v]) == self.is_greater_then[v])
        return np.where(np.array(result_vec), "tp", 'fp')


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
            if self.transformer is not None:
                result[(~mask) & (gvecs[i])] = self.models[g].predict(
                    self.transformer.fit_transform(apply_df[apply_df[self.group_column] == g]))
            else:
                result[(~mask) & (gvecs[i])] = self.models[g].predict(
                    apply_df[apply_df[self.group_column] == g])

        return result


def find_thresholds(concordance: pd.DataFrame, classify_column: str = 'classify') -> pd.DataFrame:
    quals = np.linspace(0, 2000, 30)
    sors = np.linspace(0, 20, 80)
    results = []
    pairs = []
    selection_functions = get_training_selection_functions()
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
    results = pd.concat(results, axis=1)
    results = results.T
    results.columns = results.columns.to_flat_index()

    for group in ['snp', 'h-indel', 'non-h-indel']:
        results[('recall', group)] = results.get(('tp', group), 0) / \
            (results.get(('tp', group), 0) + results.get(('fn', group), 0) + 1)
        results[('specificity', group)] = results.get(('tp', group), 0) / \
            (results.get(('tp', group), 0) + results.get(('fp', group), 0) + 1)
        results.index = pairs
    return results

# def calculate_threshold_model


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
        recall of the variable, specificity of the variable, indices of rows
        to calculate ROC curve on (output of `max_merits`) and dataframe to plot
        ROC curve
    '''

    recall = results[('recall', var_type)]
    specificity = results[('specificity', var_type)]
    idx = utils.max_merits(np.array(recall), np.array(specificity))
    results_plot = results.iloc[idx]
    return recall, specificity, idx, results_plot


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
        be plotted as `df.plot(('recall',g), ('specificity',g))
    '''
    groups = set([x[1] for x in results.columns])
    result = {}
    for g in groups:
        result[g] = get_r_s_i(results, g)[-1]
    return result


def calculate_threshold_model(results):
    all_groups = set([x[1] for x in results.columns])
    models = {}
    recalls_precisions = {}
    for g in all_groups:
        recall_series = results[('recall', g)]
        precision_series = results[('specificity', g)]
        qual, sor = ((recall_series - 1)**2 +
                     (precision_series - 1)**2).idxmin()
        recalls_precisions[g] = recall_series[(qual, sor)], precision_series[(qual, sor)]
        models[g] = SingleModel({'qual': qual, 'sor': sor}, {
                                'qual': True, 'sor': False})
    result = MaskedHierarchicalModel("thresholding", 'group', models)
    return result, recalls_precisions


def feature_prepare() -> sklearn_pandas.DataFrameMapper:
    '''Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    None

    '''
    default_filler = impute.SimpleImputer(strategy='constant', fill_value=0)
    transform_list = [(['sor'], default_filler),
                      (['dp'], default_filler),
                      ('qual', None),
                      ('inside_hmer_run', None),
                      ('close_to_hmer_run', None),
                      ('hmer_indel_nuc', preprocessing.LabelEncoder())]
    transformer = sklearn_pandas.DataFrameMapper(transform_list)
    return transformer


def precision_recall_of_set(input_set: pd.DataFrame, gtr_column: str, model: DecisionTreeClassifier) -> tuple:
    '''Precision recall calculation on a subset of data

    Parameters
    ----------
    input_set: pd.DataFrame
    gtr_column: str
        Name of the column to calculate precision-recall on
    model: DecisionTreeClassifier
        Trained model
    '''

    features = feature_prepare().fit_transform(input_set)
    gtr_values = input_set[gtr_column]
    fns = np.array(gtr_values == 'fn')
    if features[~fns, :].shape[0] > 0:
        predictions = model.predict(features[~fns, :])
    else:
        predictions = np.array([])
    predictions = np.concatenate((predictions, ['fp'] * fns.sum()))
    gtr_values = np.concatenate((gtr_values[~fns], ['tp'] * fns.sum()))

    recall = metrics.recall_score(gtr_values, predictions, pos_label='tp')
    precision = metrics.precision_score(
        gtr_values, predictions, pos_label='tp')
    return (precision, recall, predictions, gtr_values)


def calculate_predictions_gtr(concordance: pd.DataFrame, model: DecisionTreeClassifier,
                              test_train_split: pd.Series, selection: Callable, gtr_column: str) -> tuple:
    '''Returns predictions for a model

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    model: DecisionTreeClassifier
        Trained classifier
    test_train_split: pd.Series
        Split between the training and the testing set (boolean, 1 is train)
    selection: pd.Series
        Boolean series that mark specific selected rows in the concordance datafraeme
    gtr_column: str
        Name of the column

    Returns
    -------
    tuple
        train predictions, train gtr, test predictions, test gtr
    '''

    test_set = concordance[selection(concordance) & (~test_train_split)]
    test_set_predictions_gtr = precision_recall_of_set(
        test_set, gtr_column, model)[2:]

    train_set = concordance[selection(concordance) & test_train_split]
    train_set_predictions_gtr = precision_recall_of_set(
        train_set, gtr_column, model)[2:]
    return test_set_predictions_gtr + train_set_predictions_gtr


def calculate_precision_recall(concordance: pd.DataFrame, model: DecisionTreeClassifier,
                               test_train_split: pd.Series, selection: Callable, gtr_column: str) -> pd.Series:
    '''Calculates precision and recall on a model trained on a subset of data

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    model: DecisionTreeClassifier
        Trained classifier
    test_train_split: pd.Series
        Split between the training and the testing set (boolean, 1 is train)
    selection: pd.Series
        Boolean series that mark specific selected rows in the concordance datafraeme
    gtr_column: str
        Name of the column

    Returns
    -------
    pd.Series:
        The following fields are defined:
        ```
        Three parameters for unfiltered data
        basic_recall
        basic_precision
        basic_counts
        Four parameters for the filtered data
        train_precision
        train_recall
        test_precision
        test_recall
        ```
    '''

    tmp = concordance[selection(concordance)][gtr_column].value_counts()
    basic_recall = tmp.get('tp', 0) / (tmp.get('tp', 0) + tmp.get('fn', 0) + 1)
    basic_precision = tmp.get(
        'tp', 0) / (tmp.get('tp', 0) + tmp.get('fp', 0) + 1)
    basic_counts = tmp.get('tp', 0) + tmp.get('fn', 0)

    test_set = concordance[selection(concordance) & (~test_train_split)]
    test_set_precision_recall = precision_recall_of_set(
        test_set, gtr_column, model)

    train_set = concordance[selection(concordance) & test_train_split]
    train_set_precision_recall = precision_recall_of_set(
        train_set, gtr_column, model)

    return pd.Series((basic_recall, basic_precision, basic_counts) +
                     train_set_precision_recall[:2] +
                     test_set_precision_recall[:2],
                     index=['basic_recall', 'basic_precision', 'basic_counts',
                            'train_precision', 'train_recall', 'test_precision', 'test_recall'])


def train_model(concordance: pd.DataFrame, test_train_split: np.ndarray,
                selection: pd.Series, gtr_column: str,
                transformer: sklearn_pandas.DataFrameMapper) -> RandomForestClassifier:
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
    RandomForestClassifier
        Trained classifier model
    '''
    fns = np.array(concordance[gtr_column] == 'fn')
    train_data = concordance[test_train_split & selection & (~fns)][FEATURES]
    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    train_data = transformer.transform(train_data)

    model = DecisionTreeClassifier(max_depth=7)
    model.fit(train_data, labels)

    model1 = DecisionTreeRegressor
    return model


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
    sfs.append(("INDEL", lambda x: x.indel))
    sfs.append(("Non-hmer INDEL", lambda x: x.indel & (x.hmer_indel_length == 0)))
    sfs.append(("HMER indel < 4", (lambda x: x.indel & (x.hmer_indel_length > 0) &
                                                       (x.hmer_indel_length < 5))))
    sfs.append(("HMER indel > 4, < 12", lambda x: x.indel & (x.hmer_indel_length >= 5) &
                (x.hmer_indel_length < 12)))
    sfs.append(("HMER indel > 12", lambda x: x.indel & (x.hmer_indel_length >= 12)))
    return dict(sfs)


def add_testing_train_split_column(concordance: pd.DataFrame,
                                   training_groups_column: str, test_train_split_column: str,
                                   gtr_column: str,
                                   min_test_set: int = 2000, max_train_set: int = 200000,
                                   test_set_fraction: float = .2) -> pd.DataFrame:
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
        Default - 0.2

    Returns
    -------
    pd.DataFrame
        Dataframe with 0/1 in test_train_split_column, with 1 for train
    '''
    groups = set(concordance[training_groups_column])

    test_train_split_vector = np.zeros(concordance.shape[0], dtype=np.bool)
    for g in groups:
        group_vector = (concordance[training_groups_column] == g) & (concordance[gtr_column] != 'fn')
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
        assert(train_set_size / (group_vector.sum()) > test_set_fraction), \
            f"Train set fraction too small -> test:{test_set_size}, train:{train_set_size}"
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
    models = {}
    for g in groups:
        models[g] = train_model(concordance, concordance['test_train_split'],
                                concordance['group'], classify_column, transformer)

    return MaskedHierarchicalModel("Decision tree model", "group", models, transformer=transformer), concordance


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
                 (concordance[classify_column] != 'fn') & \
                 (~concordance["test_train_split"])
        group_ground_truth = concordance.loc[select, classify_column]
        group_ground_truth[group_ground_truth == 'fn'] = 'tp'
        group_predictions = predictions[select]
        recall = metrics.recall_score(group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        precision = metrics.precision_score(group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        recalls_precisions[g] = (recall, precision)

    return recalls_precisions

def train_decision_tree_regressor(concordance: pd.DataFrame, classify_column: str) -> tuple:
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
    models = {}
    for g in groups:
        models[g] = train_regression_model(concordance, concordance['test_train_split'],
                                concordance['group'], classify_column, transformer)

    return MaskedHierarchicalModel("Decision tree model", "group", models, transformer=transformer), concordance


def test_decision_tree_regressor(concordance: pd.DataFrame, model: MaskedHierarchicalModel, classify_column: str) -> dict:
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
                 (concordance[classify_column] != 'fn') & \
                 (~concordance["test_train_split"])
        group_ground_truth = concordance.loc[select, classify_column]
        group_ground_truth[group_ground_truth == 'fn'] = 'tp'
        group_predictions = predictions[select]
        recall = metrics.recall_score(group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        precision = metrics.precision_score(group_ground_truth, group_predictions, labels=["tp"], average=None)[0]
        recalls_precisions[g] = (recall, precision)

    return recalls_precisions

