from __future__ import annotations

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
    df_coverage_histogram = (
        pd.concat(
            (pd.DataFrame(json_data["base_coverage"][key], columns=[key]) for key in json_data["base_coverage"].keys()),
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    df_coverage_histogram.index.name = "coverage"
    _, df_stats = generate_stats_from_histogram(df_coverage_histogram / df_coverage_histogram.sum())
    df_stats["metric"] = df_stats.index
    df_stats_coverage = df_stats.melt(id_vars="metric")
    df_stats_coverage = df_stats_coverage.set_index(["variable", "metric"])
    df_stats_coverage.index.names = [None, None]
    df_stats_coverage.columns = [0]
    df_stats_coverage = df_stats_coverage.T
    h5_dict["stats_coverage"] = df_stats_coverage

    # add the 5x coverage percentage to RawWgsMetrics
    x5_coverage = df_coverage_histogram.loc[5:, "Genome"].sum() / df_coverage_histogram["Genome"].sum() * 100
    h5_dict["RawWgsMetrics"]["PCT_5X"] = x5_coverage

    # order the keys in RawWgsMetrics
    key_order = [
        "MEAN_COVERAGE",
        "MEDIAN_COVERAGE",
        "PCT_1X",
        "PCT_5X",
        "PCT_10X",
        "PCT_20X",
        "PCT_50X",
        "PCT_100X",
        "PCT_500X",
        "PCT_1000X",
        "FOLD_80_BASE_PENALTY_AT_30X",
        "FOLD_90_BASE_PENALTY_AT_30X",
        "FOLD_95_BASE_PENALTY_AT_30X",
    ]
    h5_dict["RawWgsMetrics"] = {k: h5_dict["RawWgsMetrics"][k] for k in key_order}

    # write to h5 file
    base_file_name = os.path.basename(input_csv_file).split(".")[0]
    if output_dir is None:
        output_dir = os.path.dirname(input_csv_file)
    output_h5_file = pjoin(output_dir, base_file_name + ".aggregated_metrics.h5")
    for key, val in h5_dict.items():
        if isinstance(val, dict):
            val = pd.DataFrame(val, index=[0])
        val.to_hdf(output_h5_file, key=key, mode="a")

    return output_h5_file
