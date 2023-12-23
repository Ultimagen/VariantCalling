from __future__ import annotations

import collections

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame

from ugvc import logger
from ugvc.filtering import variant_filtering_utils


def read_hdf(
    file_name: str,
    key: str = "all",
    skip_keys: list[str] | None = None,
    columns_subset: list[str] | None = None,
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
    columns_subset:
        select a subset of columns

    Returns
    -------
    data-frame or concat data-frame read from the h5 file according to key
    """
    if skip_keys is None:
        skip_keys = []
    if key == "all":
        with h5py.File(file_name, "r") as h5_file:
            keys = list(h5_file.keys())
        for k in skip_keys:
            if k in keys:
                keys.remove(k)
        dfs = []
        for k in keys:
            tmpdf = pd.read_hdf(file_name, key=k)
            if columns_subset is not None:
                tmpdf = tmpdf[[x for x in columns_subset if x in tmpdf.columns]]
            if tmpdf.shape[0] > 0:
                dfs.append(tmpdf)
        return pd.concat(dfs)
    if key == "all_human_chrs":
        dfs = [pd.read_hdf(file_name, key=f"chr{x}") for x in list(range(1, 23)) + ["X", "Y"]]
        return pd.concat(dfs)
    if key == "all_hg19_human_chrs":
        dfs = [pd.read_hdf(file_name, key=list(range(1, 23)) + ["X", "Y"])]
        return pd.concat(dfs)
    if key == "all_somatic_chrs":
        dfs = [pd.read_hdf(file_name, key=f"chr{x}") for x in list(range(1, 23))]
        return pd.concat(dfs)
    # If not one of the special keys:
    return pd.read_hdf(file_name, key=key)


def get_h5_keys(file_name: str) -> list[str]:
    """
    Parameters
    ----------
    file_name : str
        path to local h5 file

    Returns
    -------
    list of keys in h5 file
    """
    h5_file = h5py.File(file_name, "r")
    keys = list(h5_file.keys())
    h5_file.close()
    return keys


def calc_accuracy_metrics(
    df: DataFrame,
    classify_column_name: str,
    ignored_filters: collections.abc.Iterable[str] = (),
    group_testing_column_name: str | None = None,
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

    df = validate_preprocess_concordance(df, group_testing_column_name)
    trivial_classifier_set = initialize_trivial_classifier(ignored_filters)
    # calc recall,precision, f1 per variant category
    accuracy_df = variant_filtering_utils.eval_decision_tree_model(
        df, trivial_classifier_set, classify_column_name, group_testing_column_name is None
    )

    # Add summary for indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    all_indels = variant_filtering_utils.eval_decision_tree_model(
        df_indels,
        trivial_classifier_set,
        classify_column_name,
        add_testing_group_column=False,
    )

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='INDELS'")
    if all_indels.shape[0] == 0:
        all_indels = pd.concat(
            (all_indels, pd.DataFrame(variant_filtering_utils.get_empty_recall_precision("INDELS"), index=[0])),
            ignore_index=True,
        )
    accuracy_df = pd.concat((accuracy_df, all_indels), ignore_index=True)

    # Add summary for h-indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["hmer_indel_length"] > 0, "H-INDELS", "SNP")
    all_indels = variant_filtering_utils.eval_decision_tree_model(
        df_indels,
        trivial_classifier_set,
        classify_column_name,
        add_testing_group_column=False,
    )

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='H-INDELS'")
    if all_indels.shape[0] == 0:
        all_indels.append(
            variant_filtering_utils.get_empty_recall_precision("H-INDELS"),
            ignore_index=True,
        )
    accuracy_df = pd.concat((accuracy_df, all_indels), ignore_index=True)

    accuracy_df = accuracy_df.round(5)

    return accuracy_df


def calc_recall_precision_curve(
    df: DataFrame,
    classify_column_name: str,
    ignored_filters: collections.abc.Iterable[str] = None,
    group_testing_column_name: str | None = None,
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

    if ignored_filters is None:
        ignored_filters = ()

    df = validate_preprocess_concordance(df, group_testing_column_name)
    trivial_classifier_set = initialize_trivial_classifier(ignored_filters)
    recall_precision_curve_df = variant_filtering_utils.get_decision_tree_pr_curve(
        df, trivial_classifier_set, classify_column_name, group_testing_column_name is None
    )

    # Add summary for indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    all_indels = variant_filtering_utils.get_decision_tree_pr_curve(
        df_indels,
        trivial_classifier_set,
        classify_column_name,
        add_testing_group_column=False,
    )

    # fix boundary case when there are no indels
    all_indels = all_indels.query("group=='INDELS'")
    if all_indels.shape[0] == 0:
        pd.concat((all_indels, variant_filtering_utils.get_empty_recall_precision_curve("INDELS")), ignore_index=True)
    recall_precision_curve_df = pd.concat((recall_precision_curve_df, all_indels), ignore_index=True)

    return recall_precision_curve_df


def validate_preprocess_concordance(df: DataFrame, group_testing_column_name: str | None = None) -> DataFrame:
    """
    prepare concordance data-frame for accuracy assessment or fail if it's not possible to do

    Parameters
    ----------
    df: DataFrame
        concordance data-frame
    group_testing_column_name:
        name of column to use later for group_testing
    """
    assert "tree_score" in df.columns, "Input concordance file should be after applying a model"
    df.loc[pd.isnull(df["hmer_indel_nuc"]), "hmer_indel_nuc"] = "N"
    if np.any(pd.isnull(df["filter"])):
        logger.warning(
            "Null values in filter column (n=%i). Setting them as PASS, but it is suspicious",
            pd.isnull(df["filter"]).sum(),
        )
        df.loc[pd.isnull(df["filter"]), "filter"] = "PASS"
    if np.any(pd.isnull(df["tree_score"])):
        logger.warning(
            "Null values in concordance dataframe tree_score (n=%i). Setting them as zero, but it is suspicious",
            pd.isnull(df["tree_score"]).sum(),
        )
        df.loc[pd.isnull(df["tree_score"]), "tree_score"] = 0

    # set for compatability with eval_decision_tree_model
    df["group"] = "all"
    df["test_train_split"] = False

    # add non-default group_testing column
    if group_testing_column_name is not None:
        df["group_testing"] = df[group_testing_column_name]
        removed = pd.isnull(df["group_testing"])
        logger.info("Removing %i/%i variants with no type", removed.sum(), df.shape[0])
        df = df[~removed]
    return df


def initialize_trivial_classifier(
    ignored_filters: collections.abc.Iterable[str],
) -> variant_filtering_utils.MaskedHierarchicalModel:
    """
    initialize a classifier that will be used to simply apply filter column on the variants

    Returns
    -------
    A MaskedHierarchicalModel object representing trivial classifier which applied filter column to the variants
    """
    trivial_classifier = variant_filtering_utils.SingleTrivialClassifierModel(ignored_filters)
    trivial_classifier_set = variant_filtering_utils.MaskedHierarchicalModel(
        _name="classifier",
        _group_column="group",
        _models_dict={"all": trivial_classifier},
    )
    return trivial_classifier_set
