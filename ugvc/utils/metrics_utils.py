from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from ugvc import logger
from ugvc.comparison.concordance_utils import read_hdf


def get_h5_keys(h5_filename: str):
    with pd.HDFStore(h5_filename, "r") as store:
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


def convert_h5_to_json(
    input_h5_filename: str,
    root_element: str,
    ignored_h5_key_substring: str = None,
    output_json: str = None,
):
    """Convert an .h5 metrics file to .json with control over the root element and the processing

    Parameters
    ----------
    input_h5_filename: str
        Input h5 file name

    root_element: str
        Root element of the returned json

    ignored_h5_key_substring: str, optional
        A way to filter some of the keys using substring match, if None (default) none are filtered

    output_json : str, optional
        Output json file name to create if not None (default)

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
        df = read_hdf(input_h5_filename, h5_key)
        preprocess_columns(df)
        df_to_json = df.to_json(orient="table")
        json_dict = json.loads(df_to_json)
        new_json_dict[root_element][preprocess_h5_key(h5_key)] = json_dict

    if output_json:
        with open(output_json, "w", encoding="utf-8") as outfile:
            json.dump(new_json_dict, outfile, indent=4)
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


def read_sorter_statistics_csv(sorter_stats_csv: str, edit_metric_names: bool = True) -> pd.DataFrame:
    """
    Collect sorter statistics from csv

    Parameters
    ----------
    sorter_stats_csv : str
        path to a Sorter stats file
    edit_metric_names: bool
        if True, edit the metric names to be more human-readable

    Returns
    -------
    sorter stats as a pandas DataFrame
    """

    # read Sorter stats
    df_sorter_stats = pd.read_csv(sorter_stats_csv, header=None, names=["metric", "value"]).set_index("metric")
    # replace '(' and ')' in values (legacy format for F95)
    df_sorter_stats = df_sorter_stats.assign(
        value=df_sorter_stats["value"]
        .astype(str)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .astype(float)
        .values
    )
    # add convenient metric
    if "Failed_QC_reads" in df_sorter_stats.index and "PF_Barcode_reads" in df_sorter_stats.index:
        df_sorter_stats.loc["PCT_Failed_QC_reads"] = (
            100
            * df_sorter_stats.loc["Failed_QC_reads"]
            / (df_sorter_stats.loc["Failed_QC_reads"] + df_sorter_stats.loc["PF_Barcode_reads"])
        )

    if edit_metric_names:
        # rename metrics to uniform convention
        df_sorter_stats = df_sorter_stats.rename({c: c.replace("PCT_", "% ") for c in df_sorter_stats.index})
    return df_sorter_stats


def read_effective_coverage_from_sorter_json(
    sorter_stats_json, min_coverage_for_fp=20, max_coverage_percentile=0.95, min_mapq=60
):
    """
    Read effective coverage metrics from sorter JSON file - mean coverage, ratio of reads over MAPQ, ratio of bases in
    the given coverage range, min coverage for FP calculation, coverage of max percentile.

    Parameters
    ----------
    sorter_stats_json : str
        Path to Sorter statistics JSON file.
    min_coverage_for_fp : int
        Minimum coverage to consider for FP calculation.
    max_coverage_percentile : float
        Maximum coverage percentile to consider for FP calculation.
    min_mapq : int
        Minimum MAPQ for reads to be included

    Returns
    -------
    tuple
        (mean_coverage, ratio_of_reads_over_mapq, ratio_of_bases_in_coverage_range,
         min_coverage_for_fp, coverage_of_max_percentile)

    """
    with open(sorter_stats_json, encoding="utf-8") as fh:
        sorter_stats = json.load(fh)

    # Calculate ratio_of_bases_in_coverage_range
    cvg = pd.Series(sorter_stats["cvg"])
    cvg_cdf = cvg.cumsum() / cvg.sum()
    ratio_below_min_coverage = cvg_cdf.loc[min_coverage_for_fp]

    if ratio_below_min_coverage > 0.5:
        min_coverage_for_fp = (cvg_cdf >= 0.5).argmax()
    coverage_of_max_percentile = (cvg_cdf >= max_coverage_percentile).argmax()

    ratio_of_bases_in_coverage_range = cvg_cdf[coverage_of_max_percentile] - cvg_cdf[min_coverage_for_fp]

    # Calculate ratio_of_reads_over_mapq
    reads_by_mapq = pd.Series(sorter_stats["mapq"])
    ratio_of_reads_over_mapq = reads_by_mapq[reads_by_mapq.index >= min_mapq].sum() / reads_by_mapq.sum()

    # Calculate mean coverage
    cvg = pd.Series(sorter_stats["base_coverage"].get("Genome", sorter_stats["cvg"]))
    mean_coverage = (cvg.index.values * cvg.values).sum() / cvg.sum()

    return (
        mean_coverage,
        ratio_of_reads_over_mapq,
        ratio_of_bases_in_coverage_range,
        min_coverage_for_fp,
        coverage_of_max_percentile,
    )


def get_histogram_from_sorter(sorter_stats_json: str, histogram_key: str) -> pd.DataFrame:
    """
    Read histogram from sorter JSON file.

    Parameters
    ----------
    sorter_stats_json : str
        Path to Sorter statistics JSON file.
    histogram_key : str
        Histogram key to read from sorter JSON file.
        Allowed values: 'bqual', 'pf_bqual', 'read_length', 'aligned_read_length', 'mapq', 'cvg', 'bqx'

    Returns
    -------
    pd.DataFrame
        Histogram data.

    Raises
    ------
    ValueError
        If histogram_key is not one of the allowed values.

    """
    allowed_values = (
        "bqual",
        "pf_bqual",
        "read_length",
        "aligned_read_length",
        "mapq",
        "cvg",
        "bqx",
    )  # known histogram keys in sorter JSON
    if histogram_key not in allowed_values:
        raise ValueError(f"histogram_key must be one of {allowed_values}, got {histogram_key}")
    with open(sorter_stats_json, encoding="utf-8") as fh:
        sorter_stats = json.load(fh)
    histogram = pd.Series(sorter_stats[histogram_key], name="count")
    histogram.index.name = histogram_key

    return histogram


def merge_trimmer_histograms(trimmer_histograms: list[str], output_path: str):
    """
    Merge multiple Trimmer histograms into a single histogram.

    Parameters
    ----------
    trimmer_histograms : list[str]
        List of paths to Trimmer histogram files, or a single path to a Trimmer histogram file.
    output_path : str
        Path to output file, or a path to which the output file will be written to with the basename of the first
        value in trimmer_histograms.

    Returns
    -------
    str
        Path to output file. If the list is only 1 file, returns the path to that file without doing anything.

    Raises
    ------
    ValueError
        If trimmer_histograms is empty.
    """
    if len(trimmer_histograms) == 0:
        raise ValueError("trimmer_histograms must not be empty")
    if isinstance(trimmer_histograms, str):
        trimmer_histograms = [trimmer_histograms]
    if len(trimmer_histograms) == 1:
        return trimmer_histograms[0]

    # read and merge histograms
    df_concat = pd.concat((pd.read_csv(x) for x in trimmer_histograms))
    assert df_concat.columns[-1] == "count", f"Unexpected columns in histogram files: {df_concat.columns}"
    df_merged = df_concat.groupby(df_concat.columns[:-1].tolist(), dropna=False).sum().reset_index()
    # write to file
    output_filename = (
        os.path.join(output_path, os.path.basename(trimmer_histograms[0]))
        if os.path.isdir(output_path)
        else output_path
    )
    df_merged.to_csv(output_filename, index=False)
    return output_filename

def read_trimmer_failure_codes(trimmer_failure_codes_csv: str, add_total: bool = False) -> pd.DataFrame:
    """
    Read a trimmer failure codes csv file

    Parameters
    ----------
    trimmer_failure_codes_csv : str
        path to a Trimmer failure codes file

    Returns
    -------
    pd.DataFrame
        dataframe with trimmer failure codes

    Raises
    ------
    AssertionError
        If the columns are not as expected
    """
    df_trimmer_failure_codes = pd.read_csv(trimmer_failure_codes_csv)
    expected_columns = [
        "read group",
        "code",
        "format",
        "segment",
        "reason",
        "failed read count",
        "total read count",
    ]
    if list(df_trimmer_failure_codes.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns in {trimmer_failure_codes_csv}, expected {expected_columns}, got {list(df_trimmer_failure_codes.columns)}"
        )

    df_trimmer_failure_codes = (
        df_trimmer_failure_codes.groupby(["segment", "reason"])
        .agg({x: "sum" for x in ("failed read count", "total read count")})
        .assign(**{"% failure": lambda x: 100 * x["failed read count"] / x["total read count"]})
    )

    if add_total:
        total_row = pd.DataFrame({
            'failed read count': df_trimmer_failure_codes['failed read count'].sum(),
            'total read count': df_trimmer_failure_codes['total read count'].iloc[0],
            '% failure': df_trimmer_failure_codes['% failure'].sum()
        }, index=pd.MultiIndex.from_tuples([('total', 'total')]))

        df_trimmer_failure_codes = pd.concat([df_trimmer_failure_codes, total_row])
        df_trimmer_failure_codes.index = df_trimmer_failure_codes.index.set_names(["segment", "reason"])
        
    return df_trimmer_failure_codes