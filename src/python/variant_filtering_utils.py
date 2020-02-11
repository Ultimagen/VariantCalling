from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn import preprocessing  # type: ignore
from sklearn import metrics
import pandas as pd
import numpy as np
import tqdm
from typing import Callable, Optional
import python.utils as utils


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
    def __init__(self, _name: str, _group_column: str, _models_dict: dict):
        self.name = _name
        self.group_column = _group_column
        self.models = _models_dict

    def predict(self, df: pd.DataFrame, mask_column: Optional[str]=None) -> pd.Series:
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
            result[(~mask) & (gvecs[i])] = self.models[g].predict(
                apply_df[apply_df[self.group_column] == g])

        return result


def find_thresholds(concordance: pd.DataFrame, classify_column: str = 'classify') -> pd.DataFrame:
    quals = np.linspace(0, 2000, 30)
    sors = np.linspace(0, 20, 80)
    results = []
    pairs = []
    selection_functions = get_training_selection_functions
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


FEATURES = ['sor', 'dp', 'qual', 'hmer_indel_nuc',
            'inside_hmer_run', 'close_to_hmer_run']


def feature_prepare(df: pd.DataFrame):
    '''Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe, only features
    feature:vec: list
        List of features
    '''

    encode = preprocessing.LabelEncoder()
    if 'hmer_indel_nuc' in df.columns:
        df.loc[(df['hmer_indel_nuc']).isnull(), 'hmer_indel_nuc'] = 'N'
        df.loc[:, 'hmer_indel_nuc'] = encode.fit_transform(
            np.array(df.loc[:, 'hmer_indel_nuc']))
    if 'qd' in df.columns:
        df.loc[df['qd'].isnull(), 'qd'] = 0
    if 'sor' in df.columns:
        df.loc[df['sor'].isnull(), 'sor'] = 0
    if 'dp' in df.columns:
        df.loc[df['dp'].isnull(), 'dp'] = 0
    return df


def precision_recall_of_set(input_set: pd.DataFrame, gtr_column: str, model: RandomForestClassifier) -> tuple:
    '''Precision recall calculation on a subset of data

    Parameters
    ----------
    input_set: pd.DataFrame
    gtr_column: str
        Name of the column to calculate precision-recall on
    model: RandomForestClassifier
        Trained model
    '''

    features = feature_prepare(input_set[FEATURES])
    gtr_values = input_set[gtr_column]
    fns = np.array(gtr_values == 'fn')
    if features[~fns].shape[0] > 0:
        predictions = model.predict(features[~fns])
    else:
        predictions = np.array([])
    predictions = np.concatenate((predictions, ['fp'] * fns.sum()))
    gtr_values = np.concatenate((gtr_values[~fns], ['tp'] * fns.sum()))

    recall = metrics.recall_score(gtr_values, predictions, pos_label='tp')
    precision = metrics.precision_score(
        gtr_values, predictions, pos_label='tp')
    return (precision, recall, predictions, gtr_values)


def calculate_predictions_gtr(concordance: pd.DataFrame, model: RandomForestClassifier,
                              test_train_split: pd.Series, selection: Callable, gtr_column: str) -> tuple:
    '''Returns predictions for a model

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    model: RandomForestClassifier
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


def calculate_precision_recall(concordance: pd.DataFrame, model: RandomForestClassifier,
                               test_train_split: pd.Series, selection: Callable, gtr_column: str) -> pd.Series:
    '''Calculates precision and recall on a model trained on a subset of data

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    model: RandomForestClassifier
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
                selection: pd.Series, gtr_column: str) -> RandomForestClassifier:
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

    Returns
    -------
    RandomForestClassifier
        Trained classifier model
    '''
    fns = np.array(concordance[gtr_column] == 'fn')
    train_data = concordance[test_train_split & selection & (~fns)][FEATURES]
    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    train_data = feature_prepare(train_data)
    model = DecisionTreeClassifier(max_depth=7)
    model.fit(train_data, labels)
    return model


def train_models(concordance: pd.DataFrame, gtr_column: str,
                 selection_functions=[lambda x: np.ones(x.shape[0], dtype=np.bool)]) -> tuple:
    test_train_split = np.random.uniform(0, 1, size=concordance.shape[0]) > 0.8
    models = []
    for sf in selection_functions:
        selection = sf(concordance)
        model = train_model(concordance, test_train_split,
                            selection, gtr_column)
        models.append(model)

    return test_train_split, models


def get_training_selection_functions():
    '''
    '''
    sfs = []
    names = []
    sfs.append(lambda x: ~x.indel)
    names.append("snp")
    sfs.append(lambda x: x.indel & (x.hmer_indel_length == 0))
    names.append("non-h-indel")
    sfs.append(lambda x: x.indel & (x.hmer_indel_length > 0))
    names.apped("h-indel")
    return dict(zip(names, sfs))

def add_grouping_column(df, pd.DataFrame, selection_functions: dict, column_name: str) -> pd.DataFrame:
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
        df[selection_functions[k](df)] = k
    return df


def get_testing_selection_functions():
    sfs = []
    sfs.append((lambda x: ~x.indel, 'SNP'))
    sfs.append((lambda x: x.indel, "INDEL"))
    sfs.append((lambda x: x.indel & (x.hmer_indel_length == 0), "Non-hmer INDEL"))
    sfs.append((lambda x: x.indel & (x.hmer_indel_length > 0) &
                (x.hmer_indel_length < 5), "HMER indel < 4"))
    sfs.append((lambda x: x.indel & (x.hmer_indel_length >= 5) &
                (x.hmer_indel_length < 12), "HMER indel > 4, < 12"))
    sfs.append((lambda x: x.indel & (
        x.hmer_indel_length >= 12), "HMER indel > 12"))
    return sfs

def calculate_decision_tree_model(concordance: pd.DataFrame) -> MaskedHierarchicalModel:
    '''Train a decision tree model on the dataframe

    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe

    Returns
    -------
    MaskedHierarchicalModel
    '''
    train_selection_functions = 

