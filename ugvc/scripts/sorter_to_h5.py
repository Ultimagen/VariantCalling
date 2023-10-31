from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from os.path import join as pjoin

import pandas as pd

from ugvc.pipelines.coverage_analysis import generate_stats_from_histogram
from ugvc.utils.metrics_utils import read_sorter_statistics_csv


def sorter_to_h5(
    input_csv_file: str,
    input_json_file: str,
    metric_mapping_file: str,
    output_dir: str,
) -> str:
    """
    Aggregate sorter metrics into an h5 file of the format: aggregated_metrics.h5
    Parameters
    ----------
    input_csv_file:
        path to the sorter statistics csv file
    input_json_file:
        path to the sorter statistics json file
    metric_mapping_file:
        path to the metric mapping file,
        default: VariantCalling/ugvc/reports/sorter_output_to_aggregated_metrics_h5.csv
    output_dir:
        path to the output directory

    Returns
    -------
    output_h5_file: str
        path to the output h5 file
    """
    # Read in the input files
    input_csv = read_sorter_statistics_csv(input_csv_file, edit_metric_names=False)

    with open(input_json_file, encoding="utf-8") as jf:
        json_data = json.load(jf)
    json_keys = list(json_data.keys())

    # Read in the metric mapping file
    df_metric_mapping = pd.read_csv(metric_mapping_file)
    df_metric_mapping["H5_key"] = df_metric_mapping["H5 key, item"].str.split("/", expand=True)[0]
    df_metric_mapping["H5_item"] = df_metric_mapping["H5 key, item"].str.split("/", expand=True)[1]
    df_metric_mapping.drop(columns=["H5 key, item"], inplace=True)

    # extract metrics from the csv file
    h5_dict = defaultdict(dict)

    for _, row in df_metric_mapping.loc[df_metric_mapping["which_table"] == "csv"].iterrows():
        if row["key"] in input_csv.index:
            h5_dict[row["H5_key"]][row["H5_item"]] = input_csv.loc[row["key"], "value"]

    # extract metrics from the json file
    for _, row in df_metric_mapping.loc[df_metric_mapping["which_table"] == "json"].iterrows():
        if row["key"] in json_keys:
            h5_dict[row["H5_key"]][row["H5_item"]] = json_data[row["key"]]

    # extract coverage stats
    if not json_data["base_coverage"] == {}:  # if base_coverage is not empty
        df_coverage_histogram = (
            pd.concat(
                (
                    pd.DataFrame(json_data["base_coverage"][key], columns=[key])
                    for key in json_data["base_coverage"].keys()
                ),
                axis=1,
            )
            .fillna(0)
            .astype(int)
        )
        df_coverage_histogram.index.name = "coverage"
        df_percentiles, df_stats = generate_stats_from_histogram(
            df_coverage_histogram / df_coverage_histogram.sum(), quantiles=[0.05, 0.1, 0.2, 0.5]
        )

        df_stats["metric"] = df_stats.index
        df_stats_coverage = df_stats.melt(id_vars="metric")
        df_stats_coverage = df_stats_coverage.set_index(["variable", "metric"])
        df_stats_coverage.index.names = [None, None]
        df_stats_coverage.columns = [0]
        df_stats_coverage = df_stats_coverage.T
        # rename stats_coverage columns
        df_stats_coverage.rename(columns={"Exome": "Exome (WG)", "ACMG59": "ACMG59 (WG)"}, inplace=True)
        h5_dict["stats_coverage"] = df_stats_coverage

        # calculate coverage percetnages from the coverage histogram
        coverge_list_str = df_metric_mapping.loc[
            (df_metric_mapping["H5_key"] == "RawWgsMetrics")
            & df_metric_mapping["H5_item"].str.startswith("PCT_")
            & df_metric_mapping["H5_item"].str.endswith("X"),
            "key",
        ].tolist()
        coverage_list = [int(x.split("=")[1].replace("x", "")) for x in coverge_list_str]
        for coverage_int in coverage_list:
            h5_dict["RawWgsMetrics"][f"PCT_{coverage_int}X"] = (
                df_coverage_histogram.loc[coverage_int:, "Genome"].sum() / df_coverage_histogram["Genome"].sum()
            )

        # add the F80, F90, F95 metrics to RawWgsMetrics
        h5_dict["RawWgsMetrics"]["FOLD_80_BASE_PENALTY"] = (
            df_percentiles.loc["Q50", "Genome"] / df_percentiles.loc["Q20", "Genome"]
        )
        h5_dict["RawWgsMetrics"]["FOLD_90_BASE_PENALTY"] = (
            df_percentiles.loc["Q50", "Genome"] / df_percentiles.loc["Q10", "Genome"]
        )
        h5_dict["RawWgsMetrics"]["FOLD_95_BASE_PENALTY"] = (
            df_percentiles.loc["Q50", "Genome"] / df_percentiles.loc["Q5", "Genome"]
        )

    # order the keys in RawWgsMetrics
    key_order = [
        "MEAN_COVERAGE",
        "MEDIAN_COVERAGE",
        "PCT_1X",
        "PCT_5X",
        "PCT_10X",
        "PCT_20X",
        "PCT_30X",
        "PCT_40X",
        "PCT_50X",
        "PCT_60X",
        "PCT_70X",
        "PCT_80X",
        "PCT_90X",
        "PCT_100X",
        "PCT_500X",
        "PCT_1000X",
        "FOLD_80_BASE_PENALTY",
        "FOLD_90_BASE_PENALTY",
        "FOLD_95_BASE_PENALTY",
        "FOLD_80_BASE_PENALTY_AT_30X",
        "FOLD_90_BASE_PENALTY_AT_30X",
        "FOLD_95_BASE_PENALTY_AT_30X",
    ]
    h5_dict["RawWgsMetrics"] = {k: h5_dict["RawWgsMetrics"][k] for k in key_order if k in h5_dict["RawWgsMetrics"]}

    # Q20_PF_BASES and Q30_PF_BASES
    h5_dict["QualityYieldMetricsFlow"]["Q20_BASES"] = round(
        float(h5_dict["QualityYieldMetricsFlow"]["PCT_PF_Q20_BASES"])
        / 100
        * h5_dict["QualityYieldMetricsFlow"]["PF_BASES"]
    )
    h5_dict["QualityYieldMetricsFlow"]["Q30_BASES"] = round(
        float(h5_dict["QualityYieldMetricsFlow"]["PCT_PF_Q30_BASES"])
        / 100
        * h5_dict["QualityYieldMetricsFlow"]["PF_BASES"]
    )

    # Fill in RawWgsMetrics/PCT_EXC_DUPE with the value that is in DuplicationMetrics/PERCENT_DUPLICATION
    h5_dict["RawWgsMetrics"]["PCT_EXC_DUPE"] = h5_dict["DuplicationMetrics"]["PERCENT_DUPLICATION"]

    # write to h5 file
    base_file_name = os.path.splitext(os.path.basename(input_csv_file))[0]
    if output_dir is None:
        output_dir = os.path.dirname(input_csv_file)
    output_h5_file = pjoin(output_dir, base_file_name + ".aggregated_metrics.h5")
    for key, val in h5_dict.items():
        if isinstance(val, dict):
            val = pd.DataFrame(val, index=[0])
        val.to_hdf(output_h5_file, key=key, mode="a")

    return output_h5_file


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert sort output (csv, json file) into an aggregated h5 file\n", description=run.__doc__
    )
    parser.add_argument(
        "-c",
        "--input_csv_file",
        type=str,
        required=True,
        help="Path to the input csv file",
    )
    parser.add_argument(
        "-j",
        "--input_json_file",
        type=str,
        required=True,
        help="Path to the input json file",
    )
    parser.add_argument(
        "-m",
        "--metric_mapping_file",
        type=str,
        required=True,
        help="Path to the metric mapping file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Aggregate sorter metrics into an h5 file of the format: aggregated_metrics.h5"""
    args_in = __parse_args(argv)
    sorter_to_h5(
        input_csv_file=args_in.input_csv_file,
        input_json_file=args_in.input_json_file,
        metric_mapping_file=args_in.metric_mapping_file,
        output_dir=args_in.output_dir,
    )
