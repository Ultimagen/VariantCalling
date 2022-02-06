import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame

from python.pipelines import variant_filtering_utils as variant_filtering_utils
from ugvc import logger
from ugvc.stats.precision_recall import get_recall, get_precision, get_f1


def read_hdf(file_name: str, key='all') -> DataFrame:
    if key == 'all':
        f = h5py.File(file_name, 'r')
        keys = list(f.keys())
        print(f'keys in h5 file : {keys}')
        f.close()
        dfs = [pd.read_hdf(file_name, key=key) for key in keys]
        return pd.concat(dfs)
    elif key == 'all_human_chrs':
            dfs = [pd.read_hdf(file_name, key=f"chr{x}") for x in list(range(1, 23)) + ['X']]
            return pd.concat(dfs)
    else:
        return pd.read_hdf(file_name, key=key)

def calc_accuracy_metrics(df: DataFrame, classify_column: str, filter_hpol_run: bool = False) -> DataFrame:
    """
    Calculate tp, fp, fn, recall, precision, f1, before and after applying filter column
    @param df: concordance dataframe
    @param classify_column: column name which contains tp,fp,fn status before applying filter
    @param filter_hpol_run: filter variants with HPOL_RUN filter
    @return: data-frame with variant types and their scores
    """
    validate_and_preprocess_concordance_df(df, filter_hpol_run)
    trivial_classifier_set = initialize_trivial_classifier()
    # calc recall,precision, f1 per variant category
    accuracy_df = variant_filtering_utils.test_decision_tree_model(df,
                                                                   trivial_classifier_set,
                                                                   classify_column)
    all_indels = __summarize_indel_stats(accuracy_df)
    accuracy_df = accuracy_df.append(all_indels, ignore_index=True)
    accuracy_df = accuracy_df.round(5)
    return accuracy_df

def calc_recall_precision_curve(df: DataFrame, classify_column: str, filter_hpol_run: bool = False) -> DataFrame:
    """
    calc recall/precision curve
    @param df: concordance dataframe
    @param classify_column:  column name which contains tp,fp,fn status before applying filter
    @param filter_hpol_run: filter variants with HPOL_RUN filter
    @return:data-frame with variant types and their recall-precision curves
    """
    validate_and_preprocess_concordance_df(df, filter_hpol_run)
    trivial_classifier_set = initialize_trivial_classifier()
    recall_precision_curve_dict = \
        variant_filtering_utils.get_decision_tree_precision_recall_curve(df, trivial_classifier_set, classify_column)
    recall_precision_curve_df = __convert_recall_precision_dict_to_df({'analysis': recall_precision_curve_dict})
    return recall_precision_curve_df


def validate_and_preprocess_concordance_df(df: DataFrame, filter_hpol_run: bool = False) -> None:
    """
    prepare concordance data-frame for accuracy assessment or fail if it's not possible to do
    @param df: concordance data-frame
    @param filter_hpol_run: should we consider/ignore HPOL_RUN filter
    """
    assert 'tree_score' in df.columns, "Input concordance file should be after applying a model"
    df.loc[pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'
    if np.any(pd.isnull(df['tree_score'])):
        logger.warning("Null values in concordance dataframe tree_score. Setting them as zero, but it is suspicious")
        df.loc[pd.isnull(df['tree_score']), "tree_score"] = 0
    if not filter_hpol_run:
        df.loc[df[df['filter'] == 'HPOL_RUN'].index, 'filter'] = 'PASS'
    # set for compatability with test_decision_tree_model
    df['group'] = 'all'
    df['test_train_split'] = False

def initialize_trivial_classifier() -> variant_filtering_utils.MaskedHierarchicalModel:
    """
    initialize a classifier that will be used to simply apply filter column on the variants
    @return: MaskedHierarchicalModel
    """
    trivial_classifier = variant_filtering_utils.SingleTrivialClassifierModel()
    trivial_classifier_set = variant_filtering_utils.MaskedHierarchicalModel(_name='classifier',
                                                                             _group_column='group',
                                                                             _models_dict={'all': trivial_classifier})
    return trivial_classifier_set

def __convert_recall_precision_dict_to_df(recall_precision_dict):
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

def __summarize_indel_stats(accuracy_df):
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