from __future__ import annotations

from collections.abc import Iterable

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
    columns_subset: list[str], optional
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
            tmpdf: DataFrame = DataFrame(pd.read_hdf(file_name, key=k))
            if columns_subset is not None:
                tmpdf = tmpdf[[x for x in columns_subset if x in tmpdf.columns]]
            if tmpdf.shape[0] > 0:
                dfs.append(tmpdf)
        return pd.concat(dfs)
    if key == "all_human_chrs":
        dfs = [DataFrame(pd.read_hdf(file_name, key=f"chr{x}")) for x in list(range(1, 23)) + ["X", "Y"]]
        return pd.concat(dfs)
    if key == "all_hg19_human_chrs":
        dfs = [DataFrame(pd.read_hdf(file_name, key=x)) for x in list(range(1, 23)) + ["X", "Y"]]
        return pd.concat(dfs)
    if key == "all_somatic_chrs":
        dfs = [DataFrame(pd.read_hdf(file_name, key=f"chr{x}")) for x in list(range(1, 23))]
        return pd.concat(dfs)
    # If not one of the special keys:
    return DataFrame(pd.read_hdf(file_name, key=key))


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
    ignored_filters: Iterable[str] | None = None,
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

    Raises
    ------
    RuntimeError
        if the output of get_concordance_metrics is not a DataFrame, should not happen
    """
    if ignored_filters is None:
        ignored_filters = {"PASS"}

    df = validate_preprocess_concordance(df, group_testing_column_name)
    df["vc_call"] = df["filter"].apply(
        lambda x: convert_filter2call(x, ignored_filters=set(ignored_filters) | {"PASS"})
    )

    # calc recall,precision, f1 per variant category
    if group_testing_column_name is None:
        df = variant_filtering_utils.add_grouping_column(
            df, variant_filtering_utils.get_testing_selection_functions(), "group_testing"
        )
        group_testing_column_name = "group_testing"
    accuracy_df = variant_filtering_utils._init_metrics_df()
    groups = list(set(group_testing_column_name))
    for g_val in groups:
        dfselect = df[df[group_testing_column_name] == g_val]
        acc = variant_filtering_utils.get_concordance_metrics(
            dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
            dfselect["tree_score"].to_numpy(),
            dfselect[classify_column_name].to_numpy(),
            (dfselect[classify_column_name] == "fn").to_numpy(),
            return_curves=False,
        )
        if isinstance(acc, pd.DataFrame):
            acc["group"] = g_val
        else:
            raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
        accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)

    # Add summary for indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    for g_val in ("INDELS", "SNP"):
        dfselect = df[df["group_testing"] == g_val]
        acc = variant_filtering_utils.get_concordance_metrics(
            dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
            dfselect["tree_score"].to_numpy(),
            dfselect[classify_column_name].to_numpy(),
            (dfselect[classify_column_name] == "fn").to_numpy(),
            return_curves=False,
        )
        if isinstance(acc, pd.DataFrame):
            acc["group"] = g_val
        else:
            raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
        accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)

    # Add summary for h-indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["hmer_indel_length"] > 0, "H-INDELS", "SNP")

    g_val = "H-INDELS"
    dfselect = df[df["group_testing"] == g_val]
    acc = variant_filtering_utils.get_concordance_metrics(
        dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
        dfselect["tree_score"].to_numpy(),
        dfselect[classify_column_name].to_numpy(),
        (dfselect[classify_column_name] == "fn").to_numpy(),
        return_curves=False,
    )
    if isinstance(acc, pd.DataFrame):
        acc["group"] = g_val
    else:
        raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
    accuracy_df = pd.concat((accuracy_df, acc), ignore_index=True)

    accuracy_df = accuracy_df.round(5)

    return accuracy_df


def calc_recall_precision_curve(
    df: DataFrame,
    classify_column_name: str,
    ignored_filters: Iterable[str] | None = None,
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

    Raises
    ------
    RuntimeError
        if the output of get_concordance_metrics is not a DataFrame, should not happen
    """

    if ignored_filters is None:
        ignored_filters = {"PASS"}

    df = validate_preprocess_concordance(df, group_testing_column_name)
    df["vc_call"] = df["filter"].apply(
        lambda x: convert_filter2call(x, ignored_filters=set(ignored_filters) | {"PASS"})
    )

    # calc recall,precision, f1 per variant category
    if group_testing_column_name is None:
        df = variant_filtering_utils.add_grouping_column(
            df, variant_filtering_utils.get_testing_selection_functions(), "group_testing"
        )
        group_testing_column_name = "group_testing"

    recall_precision_curve_df = pd.DataFrame(columns=["group", "precision", "recall", "f1", "threshold"])

    groups = list(set(group_testing_column_name))
    for g_val in groups:
        dfselect = df[df[group_testing_column_name] == g_val]
        curve = variant_filtering_utils.get_concordance_metrics(
            dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
            dfselect["tree_score"].to_numpy(),
            dfselect[classify_column_name].to_numpy(),
            (dfselect[classify_column_name] == "fn").to_numpy(),
            return_metrics=False,
        )
        if isinstance(curve, pd.DataFrame):
            curve["group"] = g_val
        else:
            raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
        recall_precision_curve_df = pd.concat((recall_precision_curve_df, curve), ignore_index=True)

    # Add summary for indels
    df_indels = df.copy()
    df_indels["group_testing"] = np.where(df_indels["indel"], "INDELS", "SNP")
    for g_val in "INDELS", "SNP":
        dfselect = df[df["group_testing"] == g_val]
        curve = variant_filtering_utils.get_concordance_metrics(
            dfselect["vc_call"].replace({"tp": 1, "fp": 0}).to_numpy(),
            dfselect["tree_score"].to_numpy(),
            dfselect[classify_column_name].to_numpy(),
            (dfselect[classify_column_name] == "fn").to_numpy(),
            return_metrics=False,
        )
        if isinstance(curve, pd.DataFrame):
            curve["group"] = g_val
        else:
            raise RuntimeError("The output of get_concordance_metrics should be a DataFrame")
        recall_precision_curve_df = pd.concat((recall_precision_curve_df, curve), ignore_index=True)

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

    # add non-default group_testing column
    if group_testing_column_name is not None:
        df["group_testing"] = df[group_testing_column_name]
        removed = pd.isnull(df["group_testing"])
        logger.info("Removing %i/%i variants with no type", removed.sum(), df.shape[0])
        df = df[~removed]
    return df


def convert_filter2call(filter_str: str, ignored_filters: set | None = None) -> str:
    """Converts the filter value of the variant into tp (PASS or ignored_filters) or fp (other filters)
    Parameters
    ----------
    filter_str : str
        filter value of the variant
    ignored_filters : set, optional
        list of filters to ignore (call will be considered tp), default "PASS"

    Returns
    -------
    str:
        tp or fp
    """
    ignored_filters = {"PASS"}
    return "tp" if all(_filter in ignored_filters for _filter in filter_str.split(";")) else "fp"


def apply_filter(pre_filtering_classification: pd.Series, is_filtered: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    pre_filtering_classification : pd.Series
        classification to 'tp', 'fp', 'fn' before applying filter
    is_filtered : pd.Series
        boolean series denoting which rows where filtered

    Returns
    -------
    pd.Series
        classification to 'tp', 'fp', 'fn', 'tn' after applying filter
    """
    post_filtering_classification = pre_filtering_classification.copy()
    post_filtering_classification.loc[is_filtered & (post_filtering_classification == "fp")] = "tn"
    post_filtering_classification.loc[is_filtered & (post_filtering_classification == "tp")] = "fn"
    return post_filtering_classification
