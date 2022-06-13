import json
import re

import numpy as np
import pandas as pd

from ugvc import logger


def get_h5_keys(h5_filename: str):
    with pd.HDFStore(h5_filename) as store:
        keys = store.keys()
        return keys


def should_skip_h5_key(key: str, ignored_h5_key_substring: str):
    if ignored_h5_key_substring is None:
        return None
    return ignored_h5_key_substring in key


def preprocess_h5_key(key: str):
    result = key
    if result[0] == "/":
        result = result[1:]
    return result


def preprocess_columns(dataframe):
    """Handle multiIndex/ hierarchical .h5 - concatenate the columns for using it as single string in JSON."""

    def flatten_multi_index(col, separator):
        flat = separator.join(col)
        flat = re.sub(f"{separator}$", "", flat)
        return flat

    if hasattr(dataframe, "columns"):
        if isinstance(dataframe.columns, pd.core.indexes.multi.MultiIndex):
            dataframe.columns = [flatten_multi_index(col, "___") for col in dataframe.columns.values]


def convert_h5_to_json(input_h5_filename: str, root_element: str, ignored_h5_key_substring: str):
    """Convert an .h5 metrics file to .json with control over the root element and the processing

    Parameters
    ----------
    input_h5_filename: str
        Input h5 file name

    root_element: str
        Root element of the returned json

    ignored_h5_key_substring: str
        A way to filter some of the keys using substring match

    Returns
    -------
    str
        The result json string includes the schema (the types) of the metrics as well as the metrics themselves.

    """

    new_json_dict = {root_element: {}}
    h5_keys = get_h5_keys(input_h5_filename)
    for h5_key in h5_keys:
        if should_skip_h5_key(h5_key, ignored_h5_key_substring):
            logger.warning("Skipping: %s", h5_key)
            continue
        logger.info("Processing: %s", h5_key)
        df = pd.read_hdf(input_h5_filename, h5_key)
        preprocess_columns(df)
        df_to_json = df.to_json(orient="table")
        json_dict = json.loads(df_to_json)
        new_json_dict[root_element][preprocess_h5_key(h5_key)] = json_dict

    json_string = json.dumps(new_json_dict, indent=4)
    return json_string


def parse_md_file(md_file):
    """Parses mark duplicate Picard output"""
    with open(md_file, encoding="ascii") as infile:
        out = next(infile)
        while not out.startswith("## METRICS CLASS\tpicard.sam.DuplicationMetrics"):
            out = next(infile)

        res = pd.read_csv(infile, sep="\t")
        return np.round(float(res["PERCENT_DUPLICATION"]) * 100, 2)


def parse_cvg_metrics(metric_file):
    """Parses Picard WGScoverage metrics file

    Parameters
    ----------
    metric_file : str
        Picard metric file

    Returns
    -------
    res1 : str
        Picard file metrics class
    res2 : pd.DataFrame
        Picard metrics table
    res3 : pd.DataFrame
        Picard Histogram output
    """
    with open(metric_file, encoding="ascii") as infile:
        out = next(infile)
        while not out.startswith("## METRICS CLASS"):
            out = next(infile)

        res1 = out.strip().split("\t")[1].split(".")[-1]
        res2 = pd.read_csv(infile, sep="\t", nrows=1)
    try:
        with open(metric_file, encoding="ascii") as infile:
            out = next(infile)
            while not out.startswith("## HISTOGRAM\tjava.lang.Integer"):
                out = next(infile)

            res3 = pd.read_csv(infile, sep="\t")
    except StopIteration:
        res3 = None
    return res1, res2, res3


def parse_alignment_metrics(alignment_file):
    """Parses Picard alignment_summary_metrics file"""
    with open(alignment_file, encoding="ascii") as infile:
        out = next(infile)
        while not out.startswith("## METRICS CLASS\tpicard.analysis.AlignmentSummaryMetrics"):
            out = next(infile)

        res1 = pd.read_csv(infile, sep="\t", nrows=1)

    return res1
