import h5py
import pandas as pd
import numpy as np
from pandas import DataFrame

from python.modules.stats.precision_recall import get_recall, get_precision, get_f1
from python.pipelines import variant_filtering_utils as variant_filtering_utils
from ugvc import logger


def read_hdf(file_name: str, key='all') -> DataFrame:
    if key == 'all':
        f = h5py.File(file_name, 'r')
        keys = list(f.keys())
        print(f'keys in h5 file : {keys}')
        dfs = [pd.read_hdf(file_name, key=key) for key in keys]
        return pd.concat(dfs)
    else:
        return pd.read_hdf(file_name, key=key)

def filter_df(df):
    df = df[~((df.classify == "fp") & (df['filter'] == 'LOW_SCORE'))]
    return df


def calc_accuracy_metrics(df: DataFrame, classify_column: str, filter_hpol_run: bool = False) -> DataFrame:
    """
    Calculate tp, fp, fn, recall, precision, f1, before and after applying filter column
    :param df: concordance dataframe
    :param classify_column: column name which contains tp,fp,fn status before applying filter
    :param filter_hpol_run: filter variants with HPOL_RUN filter
    :return: data-frame with variant types and their scores
    """
    trivial_classifier_set = validate_and_preprocess_concordance_df(df, filter_hpol_run)
    # calc recall,precision, f1 per variant category
    accuracy_df = variant_filtering_utils.test_decision_tree_model(df,
                                                                   trivial_classifier_set,
                                                                   classify_column)
    all_indels = summarize_indel_stats(accuracy_df)
    accuracy_df = accuracy_df.append(all_indels, ignore_index=True)
    accuracy_df = accuracy_df.round(5)
    return accuracy_df

def calc_recall_precision_curve(df: DataFrame, classify_column: str, filter_hpol_run: bool = False) -> DataFrame:
    """
    calc recall/precision curve
    :param df: concordance dataframe
    :param classify_column:  column name which contains tp,fp,fn status before applying filter
    :param filter_hpol_run: filter variants with HPOL_RUN filter
    :return:data-frame with variant types and their recall-precision curves
    """
    trivial_classifier_set = validate_and_preprocess_concordance_df(df, filter_hpol_run)
    recall_precision_curve_dict = \
        variant_filtering_utils.get_decision_tree_precision_recall_curve(df, trivial_classifier_set, classify_column)
    recall_precision_curve_df = convert_recall_precision_dict_to_df({'analysis': recall_precision_curve_dict})
    return recall_precision_curve_df


def validate_and_preprocess_concordance_df(df: DataFrame, filter_hpol_run: bool = False):
    assert 'tree_score' in df.columns, "Input concordance file should be after applying a model"
    df.loc[pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'
    if np.any(pd.isnull(df['tree_score'])):
        logger.warning("Null values in concordance dataframe tree_score. Setting them as zero, but it is suspicious")
        df.loc[pd.isnull(df['tree_score']), "tree_score"] = 0
    print(df['filter'].value_counts())
    if not filter_hpol_run:
        df.loc[df[df['filter'] == 'HPOL_RUN'].index, 'filter'] = 'PASS'

    print(df['filter'].value_counts())
    # set for compatability with test_decision_tree_model
    df['group'] = 'all'
    df['test_train_split'] = False
    trivial_classifier = variant_filtering_utils.SingleTrivialClassifierModel()
    trivial_classifier_set = variant_filtering_utils.MaskedHierarchicalModel(_name='classifier',
                                                                             _group_column='group',
                                                                             _models_dict={'all': trivial_classifier})
    return trivial_classifier_set

def convert_recall_precision_dict_to_df(recall_precision_dict):
    results_vals = pd.DataFrame(recall_precision_dict).unstack().reset_index()
    results_vals.columns = ['model', 'category', 'tmp']
    results_vals.loc[pd.isnull(results_vals['tmp']), 'tmp'] = [(np.nan, np.nan, np.nan, np.nan)]
    results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[0])
    results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[1])
    results_vals['f1'] = results_vals['tmp'].apply(lambda x: x[2])
    if len(results_vals['tmp'].values[0]) > 3:
        results_vals['predictions'] = results_vals['tmp'].apply(lambda x: x[:, 3])
    results_vals.drop('tmp', axis=1, inplace=True)
    return results_vals

def summarize_indel_stats(accuracy_df):
    indel_df = accuracy_df[accuracy_df['group'].str.contains('indel') | accuracy_df['group'].str.contains('INDEL')]
    all_indels = indel_df.sum()
    all_indels['group'] = 'INDELS'
    all_indels['recall'] = get_recall(all_indels['fn'], all_indels['tp'])
    all_indels['precision'] = get_precision(all_indels['fp'], all_indels['tp'])
    all_indels['f1'] = get_f1(all_indels['precision'], all_indels['recall'])
    all_indels['initial_recall'] = get_recall(all_indels['initial_fn'], all_indels['initial_tp'])
    all_indels['initial_precision'] = get_precision(all_indels['initial_fp'], all_indels['initial_tp'])
    all_indels['initial_f1'] = get_f1(all_indels['initial_precision'], all_indels['initial_recall'])
    return all_indels