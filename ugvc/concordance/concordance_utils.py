from typing import List, Dict

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame

from python.pipelines import variant_filtering_utils as variant_filtering_utils
from ugvc import logger
from ugvc.utils.stats_utils import get_precision, get_recall, get_f1


def read_hdf(file_name: str, key: str = 'all') -> DataFrame:
    """
    Read data-frame or data-frames from an h5 file

    Parameters
    ----------
    file_name: str
        path of local file
    key: str
        hdf key to the data-frame.
        Special keys:
        1. all - read all data-frames from the file and concat them
        2. all_human_chrs - read chr1, ..., chr22, chrX keys, and concat them
        3. all_somatic_chrs - chr1, ..., chr22

    Returns
    -------
    data-frame or concat data-frame read from the h5 file according to key
    """
    if key == 'all':
        f = h5py.File(file_name, 'r')
        keys = list(f.keys())
        f.close()
        dfs = [pd.read_hdf(file_name, key=key) for key in keys]
        return pd.concat(dfs)
    elif key == 'all_human_chrs':
        dfs = [pd.read_hdf(file_name, key=f"chr{x}") for x in list(range(1, 23)) + ['X']]
        return pd.concat(dfs)
    elif key == 'all_somatic_chrs':
        dfs = [pd.read_hdf(file_name, key=f"chr{x}") for x in list(range(1, 23))]
        return pd.concat(dfs)
    else:
        return pd.read_hdf(file_name, key=key)


def get_h5_keys(file_name: str) -> List[str]:
    """
    Parameters
    ----------
    file_name : str
        path to local h5 file

    Returns
    -------
    list of keys in h5 file
    """
    f = h5py.File(file_name, 'r')
    keys = list(f.keys())
    f.close()
    return keys


def calc_accuracy_metrics(df: DataFrame, classify_column: str, filter_hpol_run: bool = False) -> DataFrame:
    """
    Parameters
    ----------
    df: DataFrame
        concordance dataframe
    classify_column: str
        column name which contains tp,fp,fn status before applying filter
    filter_hpol_run: bool
        filter variants with HPOL_RUN filter

    Returns
    -------
    data-frame with variant types and their scores
    """

    validate_and_preprocess_concordance_df(df, filter_hpol_run)
    trivial_classifier_set = initialize_trivial_classifier()
    # calc recall,precision, f1 per variant category
    accuracy_df = variant_filtering_utils.test_decision_tree_model(df,
                                                                   trivial_classifier_set,
                                                                   classify_column)

    # Add summary for indels
    df_indels = df.copy()
    df_indels['group_testing'] = np.where(df_indels['indel'], 'INDELS', 'SNP')
    all_indels = variant_filtering_utils.test_decision_tree_model(
        df_indels,
        trivial_classifier_set,
        classify_column,
        add_testing_group_column=False)

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='INDELS'")
    if all_indels.shape[0] == 0:
        all_indels.append(variant_filtering_utils.get_empty_recall_precision('INDELS'),
                          ignore_index=True)
    accuracy_df = accuracy_df.append(all_indels, ignore_index=True)
    accuracy_df = accuracy_df.round(5)
    return accuracy_df


def calc_recall_precision_curve(df: DataFrame, classify_column: str, filter_hpol_run: bool = False) -> DataFrame:
    """
    calc recall/precision curve

    Parameters
    ----------
    df: DataFrame
        concordance dataframe
    classify_column: str
        column name which contains tp,fp,fn status before applying filter
    filter_hpol_run: bool
         filter variants with HPOL_RUN filter

    Returns
    -------
    data-frame with variant types and their recall-precision curves
    """
    validate_and_preprocess_concordance_df(df, filter_hpol_run)
    trivial_classifier_set = initialize_trivial_classifier()
    recall_precision_curve_df = \
        variant_filtering_utils.get_decision_tree_precision_recall_curve(
            df, trivial_classifier_set, classify_column)
    # recall_precision_curve_df = __convert_recall_precision_dict_to_df(
    #    {'analysis': recall_precision_curve_dict})

    # Add summary for indels
    df_indels = df.copy()
    df_indels['group_testing'] = np.where(df_indels['indel'], 'INDELS', 'SNP')
    all_indels = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_indels,
        trivial_classifier_set,
        classify_column,
        add_testing_group_column=False)

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='INDELS'")
    if all_indels.shape[0] == 0:
        all_indels.append(variant_filtering_utils.get_empty_recall_precision_curve('INDELS'),
                          ignore_index=True)
    recall_precision_curve_df = recall_precision_curve_df.append(all_indels, ignore_index=True)

    return recall_precision_curve_df


def validate_and_preprocess_concordance_df(df: DataFrame, filter_hpol_run: bool = False) -> None:
    """
    prepare concordance data-frame for accuracy assessment or fail if it's not possible to do

    Parameters
    ----------
    df: DataFrame
        concordance data-frame
    filter_hpol_run: bool
        should we consider/ignore HPOL_RUN filter
    """
    assert 'tree_score' in df.columns, "Input concordance file should be after applying a model"
    df.loc[pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'
    if np.any(pd.isnull(df['tree_score'])):
        logger.warning(
            "Null values in concordance dataframe tree_score. Setting them as zero, but it is suspicious")
        df.loc[pd.isnull(df['tree_score']), "tree_score"] = 0
    if not filter_hpol_run:
        df.loc[df[df['filter'] == 'HPOL_RUN'].index, 'filter'] = 'PASS'
    # set for compatability with test_decision_tree_model
    df['group'] = 'all'
    df['test_train_split'] = False


def initialize_trivial_classifier() -> variant_filtering_utils.MaskedHierarchicalModel:
    """
    initialize a classifier that will be used to simply apply filter column on the variants

    Returns
    -------
    A MaskedHierarchicalModel object representing trivial classifier which applied filter column to the variants
    """
    trivial_classifier = variant_filtering_utils.SingleTrivialClassifierModel()
    trivial_classifier_set = variant_filtering_utils.MaskedHierarchicalModel(_name='classifier',
                                                                             _group_column='group',
                                                                             _models_dict={'all': trivial_classifier})
    return trivial_classifier_set


