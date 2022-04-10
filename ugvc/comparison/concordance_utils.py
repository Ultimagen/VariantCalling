from typing import Iterable, List, Optional

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame

from ugvc import logger
from ugvc.filtering import variant_filtering_utils


def read_hdf(
    file_name: str, key: str = "all", skip_keys: Iterable[str] = ()
) -> DataFrame:
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
    skip_keys: Iterable[str]
        collection of keys to skip from reading the H5 (e.g. concordance, input_args ... )
    Returns
    -------
    data-frame or concat data-frame read from the h5 file according to key
    """
    if key == "all":
        with h5py.File(file_name, "r") as f:
            keys = list(f.keys())
        for k in skip_keys:
            if k in keys:
                keys.remove(k)
        dfs = [pd.read_hdf(file_name, key=key) for key in keys]
        return pd.concat(dfs)
    elif key == "all_human_chrs":
        dfs = [
            pd.read_hdf(file_name, key=f"chr{x}") for x in list(range(1, 23)) + ["X"]
        ]
        return pd.concat(dfs)
    elif key == "all_somatic_chrs":
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
    f = h5py.File(file_name, "r")
    keys = list(f.keys())
    f.close()
    return keys


def calc_accuracy_metrics(
    df: DataFrame,
    classify_column_name: str,
    ignored_filters: Iterable[str] = (),
    group_testing_column_name: Optional[str] = None,
) -> DataFrame:
    """
    Parameters
    ----------
    df: DataFrame
        concordance dataframe
    classify_column_name: str
        column name which contains tp,fp,fn status before applying filter
    ignored_filters: Iterable[str]
        list of filters to ignore (the ignored filters will not be applied before calculating accuracy)
    group_testing_column_name: str
        column name to be used as grouping column (to output statistics on each group)

    Returns
    -------
    data-frame with variant types and their scores
    """

    df = validate_and_preprocess_concordance_df(df, group_testing_column_name)
    trivial_classifier_set = initialize_trivial_classifier(ignored_filters)
    # calc recall,precision, f1 per variant category
    accuracy_df = variant_filtering_utils.test_decision_tree_model(
        df, trivial_classifier_set, classify_column_name
    )

    # Add summary for indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    all_indels = variant_filtering_utils.test_decision_tree_model(
        df_indels,
        trivial_classifier_set,
        classify_column_name,
        add_testing_group_column=False,
    )

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='INDELS'")
    if all_indels.shape[0] == 0:
        all_indels.append(
            variant_filtering_utils.get_empty_recall_precision("INDELS"),
            ignore_index=True,
        )
    accuracy_df = accuracy_df.append(all_indels, ignore_index=True)
    accuracy_df = accuracy_df.round(5)
    return accuracy_df


def calc_recall_precision_curve(
    df: DataFrame,
    classify_column_name: str,
    ignored_filters: Iterable[str] = (),
    group_testing_column_name: Optional[str] = None,
) -> DataFrame:
    """
    calc recall/precision curve

    Parameters
    ----------
    df: DataFrame
        concordance dataframe
    classify_column_name: str
        column name which contains tp,fp,fn status before applying filter
    ignored_filters: Iterable[str]
        list of filters to ignore (the ignored filters will not be applied before calculating accuracy)
    group_testing_column_name: str
        column name to be used as grouping column (to output statistics on each group)

    Returns
    -------
    data-frame with variant types and their recall-precision curves
    """
    df = validate_and_preprocess_concordance_df(df, group_testing_column_name)
    trivial_classifier_set = initialize_trivial_classifier(ignored_filters)
    recall_precision_curve_df = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df, trivial_classifier_set, classify_column_name
    )

    # Add summary for indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    all_indels = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_indels,
        trivial_classifier_set,
        classify_column_name,
        add_testing_group_column=False,
    )

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='INDELS'")
    if all_indels.shape[0] == 0:
        all_indels.append(
            variant_filtering_utils.get_empty_recall_precision_curve("INDELS"),
            ignore_index=True,
        )
    recall_precision_curve_df = recall_precision_curve_df.append(
        all_indels, ignore_index=True
    )

    return recall_precision_curve_df


def validate_and_preprocess_concordance_df(
    df: DataFrame, group_testing_column_name: Optional[str]
) -> DataFrame:
    """
    prepare concordance data-frame for accuracy assessment or fail if it's not possible to do

    Parameters
    ----------
    df: DataFrame
        concordance data-frame
    group_testing_column_name:
        name of column to use later for group_testing
    """
    assert (
        "tree_score" in df.columns
    ), "Input concordance file should be after applying a model"
    df.loc[pd.isnull(df["hmer_indel_nuc"]), "hmer_indel_nuc"] = "N"
    if np.any(pd.isnull(df["filter"])):
        logger.warning(
            f"Null values in filter column (n={pd.isnull(df['filter']).sum()}). "
            f"Setting them as PASS, but it is suspicious"
        )
        df.loc[pd.isnull(df["filter"]), "filter"] = "PASS"
    if np.any(pd.isnull(df["tree_score"])):
        logger.warning(
            f"Null values in concordance dataframe tree_score (n={pd.isnull(df['tree_score']).sum()}). "
            f"Setting them as zero, but it is suspicious"
        )
        df.loc[pd.isnull(df["tree_score"]), "tree_score"] = 0

    # set for compatability with test_decision_tree_model
    df["group"] = "all"
    df["test_train_split"] = False

    # add non-default group_testing column
    if group_testing_column_name is not None:
        df["group_testing"] = df[group_testing_column_name]
        removed = pd.isnull(df["group_testing"])
        logger.info(f"Removing {removed.sum()}/{df.shape[0]} variants with no type")
        df = df[~removed]
    return df


def initialize_trivial_classifier(
    ignored_filters: Iterable[str],
) -> variant_filtering_utils.MaskedHierarchicalModel:
    """
    initialize a classifier that will be used to simply apply filter column on the variants

    Returns
    -------
    A MaskedHierarchicalModel object representing trivial classifier which applied filter column to the variants
    """
    trivial_classifier = variant_filtering_utils.SingleTrivialClassifierModel(
        ignored_filters
    )
    trivial_classifier_set = variant_filtering_utils.MaskedHierarchicalModel(
        _name="classifier",
        _group_column="group",
        _models_dict={"all": trivial_classifier},
    )
    return trivial_classifier_set
