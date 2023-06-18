from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


def get_dict_from_dataframe(
    df_input: pd.DataFrame,
    detail_type: str,
):
    """
    Goal: Function for creating dictionary for json file output files
    Receives dataframe, converts output to dictionary, which is used to print out json file of input data

    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe to be converted to dictionary
    detail_type : str
        Representing the table type derived from MethylDackel methods
    Returns
    -------
    out_dict: dict

    """
    out_dict = {}
    out_dict[detail_type] = []
    df_subset = df_input[df_input["detail"] == detail_type]

    # Step #1: -------------- get top tables of metrics --------------
    pat = "Total|^PercentMethylation_[a-zA-Z]|^Coverage_[a-zA-Z]"

    # for pat in patterns:
    idx = df_subset.metric.str.contains(pat)
    df_detail = df_subset.loc[idx, :].copy()
    df_detail.drop("detail", inplace=True, axis=1)
    temp_dict = df_detail.to_dict("records")
    out_dict[detail_type].append(temp_dict)

    # Step #2: -------------- get 2D array of metrics --------------
    if detail_type == "hg":
        patterns = ["^PercentMethylation_[0-9]", "^Coverage_[0-9]", "CumulativeCoverage_[0-9]"]
    else:
        patterns = [
            "^PercentMethylation_[0-9]",
            "^Coverage_[0-9]",
            "CumulativeCoverage_[0-9]",
            "^PercentMethylationPosition",
        ]

    cols = ["metric", "value"]  # only these columns without genome

    for pat in patterns:
        idx = df_subset.metric.str.contains(pat)
        if any(idx):
            df_detail = df_subset.loc[idx, :].copy()
            df_detail.drop("detail", inplace=True, axis=1)

            temp_dict = {
                "keys": list(["metric", "value"]),
                "values": [list(x) for x in list(df_detail[cols][["metric", "value"]].itertuples(index=False))],
            }
            out_dict[detail_type].append(temp_dict)

    return out_dict


def find_list_genomes(
    in_list: list,
):
    """
    Goal: finding the genomes in the input file
    Receives list of strings which represent the different genomes, possible genomes: hg, Lambda, pUC19

       Parameters
       ----------
       in_list: list
           Input list of strings of genome(s)

       Returns
       -------
       match: list

    """
    string = "".join(in_list)
    match = re.findall(r"chr|Lambda|pUC19", string, re.IGNORECASE)
    match = list(Counter(match))
    return match


def get_ctrl_genomes_data(
    data_frame: pd.DataFrame,
    list_genomes: list,
):
    """
    Goal: calculating percent of methylation from genomes that are not hg (i.e.: Lambda, pUC19)
    Receives dataframe and list of strings which represent the different genomes
    Returns dataframe with percent methylation of control genomes

       Parameters
       ----------
       data_frame: pd.DataFrame
            Input dataframe with methylation data

       list_genomes: list
            Input list of strings of genome(s)

       Returns
       -------
       df_output: pd.DataFrame

    """
    if len(list_genomes) > 1:
        col = ["PercentMethylation"]
        prefix = "PercentMethylationPosition_"
        col_names = ["value", "metric", "detail"]
        df_output = pd.DataFrame(columns=col_names)
        df_control = pd.DataFrame(columns=col_names)
        for temp_genome in list_genomes:
            pat = r"^" + temp_genome
            idx = data_frame.chr.str.contains(pat)
            if idx.any(axis=None):
                data_frame_sub = data_frame.loc[idx, :]
                df_control = data_frame_sub[col].copy()
                df_control.reset_index(inplace=True, drop=True)
                df_control.loc[:, "metric"] = prefix + df_control.index.astype(str)
                df_control.loc[:, "detail"] = temp_genome
                df_control.columns = col_names
                df_control = df_control[df_control.columns[[1, 0, 2]]]

            df_output = pd.concat([df_output, df_control], axis=0, ignore_index=True)

        return df_output
    return None


def calc_percent_methylation(
    table_type: str,
    data_frame: pd.DataFrame,
    rel: bool,
):
    """
    Goal: Function for calculating distribution of % methylation

    Parameters
    ----------
    table_type: str
         Represents table type derived from the various MethylDackel functions used
    data_frame: pd.DataFrame
        Input dataframe with methylation data derived from MethylDackel methods
    rel: str
        indicate if absolute data of CpGs should be outputted (occurs if rel is false)

    Returns
    -------
    df_output: pd.DataFrame
        Output is dataframe with percent methylation and optionally absolute data of CpG amount

    """

    if not data_frame.empty:
        n_pcnt = 110
        bins_pcnt = np.arange(start=0, stop=n_pcnt, step=10)

        rows = ["mean", "std", "50%"]
        col = "PercentMethylation"

        df_distrib = pd.DataFrame()

        x = data_frame[col]

        value, metric = np.histogram(x, bins=bins_pcnt)  # where bins are 0-9, 10-19.., 90-100
        metric += 10
        df_pcnt = pd.concat([pd.Series(metric), pd.Series(value)], axis=1)
        df_pcnt.columns = ["metric", "value"]  # include 100 as well

        df_pcnt.dropna(axis=0, inplace=True)
        if rel:  # get relative values
            df_pcnt.loc[:, "value"] = df_pcnt["value"] / np.sum(df_pcnt["value"])

        # add other metrics long format
        desc = x.describe()
        desc = desc[rows]
        desc = desc.rename(index={"50%": "median"})
        desc = pd.DataFrame({"metric": desc.index, "value": desc})
        df_pcnt = pd.concat([df_pcnt, desc], axis=0, ignore_index=True)
        df_pcnt.loc[:, "metric"] = col + "_" + df_pcnt["metric"].astype(str)

        if not rel:
            add_row = pd.DataFrame({"metric": "TotalCpGs", "value": np.array(x).size}, index=[0])
            df_pcnt = pd.concat([df_pcnt, add_row], axis=0, ignore_index=True)
        df_distrib = pd.concat((df_distrib, df_pcnt), ignore_index=True)
        df_distrib.fillna(0, inplace=True)
        df_distrib["detail"] = table_type
        df_distrib.drop_duplicates(inplace=True)

        return df_distrib
    return None


def calc_coverage_methylation(detail_type: str, data_frame: pd.DataFrame, rel: str):
    """
    Goal: Function for calculating distribution of CpG Coverage

    Parameters
    ----------
    detail_type: str
        Represents input type - could be genome or type of Cs examined, e.g.: CHG, CHH
    data_frame: pd.DataFrame
       Input dataframe with methylation data derived MethylDackel methods
    rel: str
       indicate if absolute data of CpGs should be outputted (occurs if rel is false)

    Returns
    -------
    df_output: pd.DataFrame
       Output is dataframe with relative coverage and cumulative coverage

    """

    if not data_frame.empty:
        # Initialise arguments for function
        n_cov = 210
        n_cap = 200

        bins_cov = np.arange(start=0, stop=n_cov, step=10)

        rows = ["mean", "std", "50%"]
        col_names = ["Coverage", "TotalCpGs"]
        df_cols = list(data_frame.columns)
        cols = list(set(col_names).intersection(set(df_cols)))

        df_distrib = pd.DataFrame()
        df_rel_cov = pd.DataFrame(columns=["metric", "value", "detail"])

        for col in cols:
            x = data_frame[col]

            x = x.mask(x > n_cap, n_cap)  # use relative values from above
            value, metric = np.histogram(x, bins=bins_cov)
            metric += 10
            df_abs_cov = pd.concat([pd.Series(metric), pd.Series(value)], axis=1)
            df_abs_cov.columns = ["metric", "value"]
            df_abs_cov.dropna(axis=0, inplace=True)

            # add other metrics to absolute values use x from above
            desc = x.describe()
            desc = desc[rows]
            desc = desc.rename(index={"50%": "median"})
            desc = pd.DataFrame({"metric": desc.index, "value": desc})

            # assign names for metrics
            df_abs_cov.loc[:, "metric"] = col + "_" + df_abs_cov["metric"].astype(str)

            # calc relative values
            if not col.startswith("TotalCpG"):
                y = np.cumsum(df_abs_cov["value"]) / (np.sum(df_abs_cov["value"]))
                df_rel_cov = df_abs_cov.copy()
                df_rel_cov.loc[:, "value"] = y
                df_rel_cov.loc[:, "metric"] = "Cumulative" + df_rel_cov["metric"].astype(str)

            if not rel:
                desc.loc[:, "metric"] = col + "_" + desc["metric"].astype(str)
                df_abs_cov = pd.concat([df_abs_cov, desc], axis=0, ignore_index=True)

            else:
                df_abs_cov = desc.copy()
                df_abs_cov.loc[:, "metric"] = col + "_" + df_abs_cov["metric"].astype(str)

            df_distrib = pd.concat((df_distrib, df_abs_cov, df_rel_cov), ignore_index=True)
            df_distrib.fillna(0, inplace=True)
            df_distrib["detail"] = detail_type

            df_distrib.drop_duplicates(inplace=True)

        return df_distrib
    return None


def calc_TotalCpGs(
    key_word: str,
    data_frame: pd.DataFrame,
):
    """
    Goal: Function for calculating distribution of total CpGs

    Parameters
    ----------
    key_word: str
        gives detail of type of calculation
    data_frame: pd.DataFrame
       Input dataframe with methylation data derived from one of MethylDackel preRead method

    Returns
    -------
    df_output: pd.DataFrame
       Output is dataframe with total CpGs

    """

    rows = ["mean", "std", "50%"]
    col = "TotalCpGs"

    df_distrib = pd.DataFrame()

    x = data_frame[col]

    desc = x.describe()
    desc = desc[rows]
    desc = desc.rename(index={"50%": "median"})
    desc = pd.DataFrame({"metric": desc.index, "value": desc})
    desc.loc[:, "metric"] = col + "_" + desc["metric"].astype(str)
    # output
    df_distrib = pd.concat((df_distrib, desc), ignore_index=True)
    df_distrib.fillna(0, inplace=True)
    df_distrib["detail"] = key_word
    df_distrib.drop_duplicates(inplace=True)

    return df_distrib


# ====================================================================
def calc_distrib_per_strand(data_frame: pd.DataFrame):
    """
    Goal: Function for calculating distribution per strand in Mbias

    Parameters
    ----------
    data_frame: pd.DataFrame
       Input dataframe with methylation data derived from one of MethylDackel preRead method

    Returns
    -------
    df_output: pd.DataFrame
       Output is dataframe with percent methylation per each position levels along reads (strands)

    """

    # initalise
    rows = ["mean", "std", "50%"]

    grouped_by_strands = data_frame.groupby(["Strand"])
    df_distrib = pd.concat([pd.DataFrame(y) for x, y in grouped_by_strands]).sort_index()

    # change and sort column names
    # --------------------------------------------
    col = "PercentMethylationPosition"
    df_distrib.columns = ["detail", "metric", "value"]
    df_distrib.loc[:, "metric"] = col + "_" + df_distrib["metric"].astype(str)
    df_distrib = df_distrib[df_distrib.columns[[1, 2, 0]]]

    # add descriptive statistics: mean, std, median
    # --------------------------------------------
    col = "PercentMethylation"
    for temp_strand in df_distrib["detail"].unique():
        df_subset = df_distrib[df_distrib["detail"] == temp_strand].copy()
        x = df_subset["value"]
        # get descriptive statistics per strand
        desc = x.describe()
        desc = desc[rows]
        desc = desc.rename(index={"50%": "median"})
        desc = pd.DataFrame({"metric": desc.index, "value": desc})
        desc.loc[:, "detail"] = temp_strand
        desc.loc[:, "metric"] = col + "_" + desc["metric"]
        df_distrib = pd.concat([df_distrib, desc], axis=0, ignore_index=True)

    return df_distrib
