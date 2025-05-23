#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Evaluate performance of a callset given its comparison to the ground truth
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse
import logging
import sys

from pandas import DataFrame

from ugvc import logger
from ugbio_core.concordance.concordance_utils import calc_accuracy_metrics, calc_recall_precision_curve
from ugbio_core.h5_utils import read_hdf
from ugbio_core.vcfbed import vcftools


def parse_args(argv: list[str]):
    ap_var = argparse.ArgumentParser(prog="evaluate_concordance.py", description=run.__doc__)
    ap_var.add_argument("--input_file", help="Name of the input h5 file", type=str, required=True)
    ap_var.add_argument("--output_prefix", help="Prefix to output files", type=str, required=True)
    ap_var.add_argument("--dataset_key", help="h5 dataset name, such as chromosome name", default="all")
    ap_var.add_argument("--score_key", help="info key name for calculating the score", default="tree_score")
    ap_var.add_argument(
        "--ignore_genotype",
        help="ignore genotype when comparing to ground-truth",
        action="store_true",
        default=False,
    )
    ap_var.add_argument(
        "--ignore_filters",
        help="comma separated list of filters to ignore",
        default="HPOL_RUN",
    )
    ap_var.add_argument(
        "--output_bed",
        help="output bed files of fp/fn/tp per variant-type",
        action="store_true",
        default=False,
    )
    ap_var.add_argument(
        "--use_for_group_testing",
        help="Column in the h5 to use for grouping (or generate default groupings)",
        type=str,
    )
    ap_var.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    args = ap_var.parse_args(argv)
    return args


def run(argv: list[str]):
    """Calculate precision and recall for compared HDF5"""
    args = parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))
    ds_key = args.dataset_key
    out_pref = args.output_prefix
    ignore_genotype = args.ignore_genotype
    ignored_filters = args.ignore_filters.split(",")
    output_bed = args.output_bed
    group_testing_column = args.use_for_group_testing

    # comparison dataframes often contain dataframes that we do not want to read
    if ds_key == "all":
        skip = ["concordance", "scored_concordance", "input_args", "comparison_result"]
    else:
        skip = []
    df: DataFrame = read_hdf(args.input_file, key=ds_key, skip_keys=skip)

    # Enable evaluating concordance from vcf without tree-score field
    score_column = args.score_key.lower()
    if score_column not in df.columns or all(df[score_column].isna()):
        df[score_column] = 1
        logger.warning(f"No {score_column} field in comparison hdf input, expect invalid recall/precision curves")

    # TODO: make selection of the scoring column cleaner. Currently tree_score is hard coded as a scoring column in
    #       multiple places, and it was too much work to convert it.
    df["tree_score"] = df[score_column]
    classify_column = "classify" if ignore_genotype else "classify_gt"

    accuracy_df = calc_accuracy_metrics(df, classify_column, ignored_filters, group_testing_column)
    accuracy_df.to_hdf(f"{out_pref}.h5", key="optimal_recall_precision")
    accuracy_df.to_csv(f"{out_pref}.stats.csv", sep=";", index=False)

    recall_precision_curve_df = calc_recall_precision_curve(df, classify_column, ignored_filters, group_testing_column)
    recall_precision_curve_df.to_hdf(f"{out_pref}.h5", key="recall_precision_curve")
    if output_bed:
        vcftools.bed_files_output(df, f"{out_pref}.h5", mode="w", create_gt_diff=True)
    recall_precision_curve_df[["group", "threshold"]].to_csv(f"{out_pref}.thresholds.csv", index=None)


if __name__ == "__main__":
    run(sys.argv[1:])
