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
#    Train ML models to filter callset
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys

import numpy as np
import pandas as pd

import ugvc.filtering.blacklist as blacklist_fcn
from ugvc import logger
from ugvc.comparison import vcf_pipeline_utils
from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.filtering import variant_filtering_utils
from ugvc.vcfbed import vcftools


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="train_models_pipeline.py",
        description="Train filtering models on the concordance file",
    )

    ap_var.add_argument(
        "--input_file",
        help="Name of the input h5/vcf file. h5 is output of comparison",
        type=str,
    )
    ap_var.add_argument(
        "--blacklist",
        help="blacklist file by which we decide variants as FP",
        type=str,
        required=False,
    )
    ap_var.add_argument(
        "--output_file_prefix",
        help="Output .pkl file with models, .h5 file with results",
        type=str,
        required=True,
    )
    ap_var.add_argument("--mutect", required=False, action="store_true")
    ap_var.add_argument(
        "--evaluate_concordance",
        help="Should the results of the model be applied to the concordance dataframe",
        action="store_true",
    )
    ap_var.add_argument(
        "--apply_model",
        help="If evaluate_concordance - which model should be applied",
        type=str,
        required="--evaluate_concordance" in sys.argv,
    )
    ap_var.add_argument(
        "--evaluate_concordance_contig",
        help="Which contig the evaluation of the model should be done on",
        default="chr9",
    )

    ap_var.add_argument(
        "--input_interval",
        help="bed file of intersected intervals from run_comparison pipeline",
        type=str,
        required=False,
    )
    ap_var.add_argument(
        "--list_of_contigs_to_read",
        nargs="*",
        help="List of contigs to read from the DF",
        default=[],
    )
    ap_var.add_argument("--reference", help="Reference genome", required=True, type=str)
    ap_var.add_argument(
        "--runs_intervals",
        help="Runs intervals (bed/interval_list)",
        required=False,
        type=str,
        default=None,
    )
    ap_var.add_argument(
        "--annotate_intervals",
        help="interval files for annotation (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )
    ap_var.add_argument(
        "--exome_weight",
        help="weight of exome variants in comparison to whole genome variant",
        type=int,
        default=1,
    )
    ap_var.add_argument(
        "--flow_order",
        help="Sequencing flow order (4 cycle)",
        required=False,
        default=DEFAULT_FLOW_ORDER,
    )
    ap_var.add_argument(
        "--exome_weight_annotation",
        help="annotation name by which we decide the weight of exome variants",
        type=str,
    )
    ap_var.add_argument(
        "--vcf_type",
        help='VCF type - "single_sample" or "joint"',
        type=str,
        default="single_sample",
    )
    ap_var.add_argument(
        "--ignore_filter_status",
        help="Ignore the `filter` and `tree_score` columns",
        action="store_true",
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
    #  pylint: disable=too-many-statements
    "Train filtering models"
    np.random.seed(1984)
    random.seed(1984)
    args = parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))

    try:
        if args.input_file.endswith("h5"):
            assert args.input_interval, "--input_interval is required when input file type is h5"
        if args.input_file.endswith("vcf.gz"):
            assert args.blacklist, "--blacklist is required when input file type is vcf.gz"
            assert args.reference, "--reference is required when input file type is vcf.gz"
            assert args.runs_intervals, "--runs_intervals is required when input file type is vcf.gz"
    except AssertionError as af_var:
        logger.error(str(af_var))
        raise af_var

    try:
        with_dbsnp_bl = args.input_file.endswith("vcf.gz")
        if with_dbsnp_bl:
            if args.list_of_contigs_to_read != []:
                dfs = [vcftools.get_vcf_df(args.input_file, chromosome=x) for x in args.list_of_contigs_to_read]
                df = pd.concat(dfs)
            else:
                df = vcftools.get_vcf_df(args.input_file)
        else:
            # read all data besides concordance and input_args or as defined in list_of_contigs_to_read
            df = []
            annots = []
            with pd.HDFStore(args.input_file) as data:
                for k in data.keys():
                    if (
                        (k != "/concordance")
                        and (k != "/input_args")
                        and (args.list_of_contigs_to_read == [] or k[1:] in args.list_of_contigs_to_read)
                    ):
                        h5_file = data.get(k)
                        if not h5_file.empty:
                            df.append(h5_file)

            df = pd.concat(df, axis=0)

        if args.ignore_filter_status:
            df["filter"] = ""
            df["tree_score"] = None

        df, annots = vcf_pipeline_utils.annotate_concordance(
            df,
            args.reference,
            runfile=args.runs_intervals,
            flow_order=args.flow_order,
            annotate_intervals=args.annotate_intervals,
        )

        if args.mutect:
            df["qual"] = df["tlod"].apply(lambda x: max(x) if isinstance(x, tuple) else 50) * 10
        df.loc[pd.isnull(df["hmer_indel_nuc"]), "hmer_indel_nuc"] = "N"

        results_dict = {}

        if with_dbsnp_bl:
            blacklist = pd.read_hdf(args.blacklist, "blacklist")
            df = df.merge(blacklist, left_index=True, right_index=True, how="left")
            df["bl"].fillna(False, inplace=True)
            df["bl_classify"] = "unknown"
            df["bl_classify"].loc[df["bl"]] = "fp"
            df["bl_classify"].loc[~df["id"].isna()] = "tp"
            classify_clm = "bl_classify"
            blacklist_statistics = blacklist_fcn.create_blacklist_statistics_table(df, classify_clm)
            df = df[df["bl_classify"] != "unknown"]
            # Decision tree models
            interval_size = None
        else:
            classify_clm = "classify"
            interval_size = vcf_pipeline_utils.bed_file_length(args.input_interval)
        # Thresholding model

        logger.debug("INTERVAL_SIZE = %i", interval_size)

        models_thr_no_gt, df_tmp = variant_filtering_utils.train_threshold_models(
            concordance=df.copy(),
            interval_size=interval_size,
            classify_column=classify_clm,
            annots=annots,
            vtype=args.vcf_type,
        )

        recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
            df_tmp, models_thr_no_gt, classify_column=classify_clm
        )

        recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_pr_curve(
            df_tmp, models_thr_no_gt, classify_column=classify_clm
        )
        df_tmp["test_train_split"] = ~df_tmp["test_train_split"]
        recall_precision_no_gt_train = variant_filtering_utils.test_decision_tree_model(
            df_tmp, models_thr_no_gt, classify_column=classify_clm
        )

        recall_precision_curve_no_gt_train = variant_filtering_utils.get_decision_tree_pr_curve(
            df_tmp, models_thr_no_gt, classify_column=classify_clm
        )

        results_dict["threshold_model_ignore_gt_incl_hpol_runs"] = models_thr_no_gt
        results_dict["threshold_model_recall_precision_ignore_gt_incl_hpol_runs"] = recall_precision_no_gt

        results_dict["threshold_model_recall_precision_curve_ignore_gt_incl_hpol_runs"] = recall_precision_curve_no_gt
        results_dict["threshold_train_model_recall_precision_ignore_gt_incl_hpol_runs"] = recall_precision_no_gt
        results_dict[
            "threshold_train_model_recall_precision_curve_ignore_gt_incl_hpol_runs"
        ] = recall_precision_curve_no_gt

        # RF model
        models_rf_no_gt, df_tmp = variant_filtering_utils.train_model_wrapper(
            df.copy(),
            classify_column=classify_clm,
            interval_size=interval_size,
            train_function=variant_filtering_utils.train_model_rf,
            model_name="Random forest",
            annots=annots,
            exome_weight=args.exome_weight,
            exome_weight_annotation=args.exome_weight_annotation,
            use_train_test_split=True,
            vtype=args.vcf_type,
        )

        recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(df_tmp, models_rf_no_gt, classify_clm)

        recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_pr_curve(
            df_tmp, models_rf_no_gt, classify_clm
        )

        df_tmp["test_train_split"] = ~df_tmp["test_train_split"]
        recall_precision_no_gt_train = variant_filtering_utils.test_decision_tree_model(
            df_tmp, models_rf_no_gt, classify_clm
        )

        recall_precision_curve_no_gt_train = variant_filtering_utils.get_decision_tree_pr_curve(
            df_tmp, models_rf_no_gt, classify_clm
        )

        results_dict["rf_model_ignore_gt_incl_hpol_runs"] = models_rf_no_gt
        results_dict["rf_model_recall_precision_ignore_gt_incl_hpol_runs"] = recall_precision_no_gt
        results_dict["rf_model_recall_precision_curve_ignore_gt_incl_hpol_runs"] = recall_precision_curve_no_gt
        results_dict["rf_train_model_recall_precision_ignore_gt_incl_hpol_runs"] = recall_precision_no_gt_train
        results_dict[
            "rf_train_model_recall_precision_curve_ignore_gt_incl_hpol_runs"
        ] = recall_precision_curve_no_gt_train

        with open(args.output_file_prefix + ".pkl", "wb") as file:
            pickle.dump(results_dict, file)

        accuracy_dfs = []
        prcdict = {}
        for m_var in ("threshold", "threshold_train", "rf", "rf_train"):
            name_optimum = f"{m_var}_model_recall_precision_ignore_gt_incl_hpol_runs"
            accuracy_df_per_model = results_dict[name_optimum]
            accuracy_df_per_model["model"] = name_optimum
            accuracy_dfs.append(accuracy_df_per_model)
            prcdict[name_optimum] = results_dict[
                name_optimum.replace("recall_precision", "recall_precision_curve")
            ].set_index("group")

        accuracy_df = pd.concat(accuracy_dfs)
        accuracy_df.to_hdf(args.output_file_prefix + ".h5", key="optimal_recall_precision")

        results_vals = pd.concat(prcdict, names=["model"])
        results_vals = results_vals[["recall", "precision", "f1"]].reset_index()

        results_vals.to_hdf(args.output_file_prefix + ".h5", key="recall_precision_curve")
        if with_dbsnp_bl:
            blacklist_statistics.to_hdf(args.output_file_prefix + ".h5", key="blacklist_statistics")

        if args.evaluate_concordance:
            if with_dbsnp_bl:
                calls_df = vcftools.get_vcf_df(args.input_file, chromosome=args.evaluate_concordance_contig)
            else:
                calls_df = pd.read_hdf(args.input_file, "concordance")

            if args.ignore_filter_status:
                calls_df["filter"] = ""
                calls_df["tree_score"] = None

            calls_df, _ = vcf_pipeline_utils.annotate_concordance(
                calls_df,
                args.reference,
                runfile=args.runs_intervals,
                annotate_intervals=args.annotate_intervals,
            )
            if args.mutect:
                calls_df["qual"] = calls_df["tlod"].apply(lambda x: max(x) if isinstance(x, tuple) else 50) * 10
            calls_df.loc[pd.isnull(calls_df["hmer_indel_nuc"]), "hmer_indel_nuc"] = "N"

            models = results_dict[args.apply_model]
            model_clsf = models

            logger.info("Applying classifier")
            predictions = model_clsf.predict(
                variant_filtering_utils.add_grouping_column(
                    calls_df,
                    variant_filtering_utils.get_training_selection_functions(),
                    "group",
                )
            )
            logger.info("Applying regressor")

            predictions_score = model_clsf.predict(
                variant_filtering_utils.add_grouping_column(
                    calls_df,
                    variant_filtering_utils.get_training_selection_functions(),
                    "group",
                ),
                get_numbers=True,
            )

            calls_df["prediction"] = predictions
            calls_df["tree_score"] = predictions_score
            # In case we already have filter column, reset the PASS,
            # Then, by the prediction of the model we decide whether the filter column is PASS or LOW_SCORE
            calls_df["filter"] = (
                calls_df["filter"]
                .apply(lambda x: x.replace("PASS;", ""))
                .apply(lambda x: x.replace(";PASS", ""))
                .apply(lambda x: x.replace("PASS", ""))
            )
            calls_df.loc[calls_df["prediction"] == "fp", "filter"] = calls_df.loc[
                calls_df["prediction"] == "fp", "filter"
            ].apply(lambda x: "LOW_SCORE" if x == "" else x + ";LOW_SCORE")
            calls_df.loc[calls_df["prediction"] == "tp", "filter"] = calls_df.loc[
                calls_df["prediction"] == "tp", "filter"
            ].apply(lambda x: "PASS" if x == "" else x + ";PASS")
            calls_df.to_hdf(f"{args.output_file_prefix}.h5", key="scored_concordance")
        logger.info("Model training run: success")

    except Exception as err:
        exc_info = sys.exc_info()
        logger.error(*exc_info)
        logger.error("Model training run: failed")
        raise err


if __name__ == "__main__":
    run(sys.argv[1:])
