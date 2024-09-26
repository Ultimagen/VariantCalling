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
import random
import sys

import dill as pickle
import numpy as np
import pandas as pd

from ugvc import logger
from ugvc.filtering import transformers, variant_filtering_utils
from ugvc.filtering.tprep_constants import GtType, VcfType
from ugbio_core.h5_utils import read_hdf


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="train_models_pipeline.py",
        description="Train filtering models",
    )

    ap_var.add_argument(
        "--train_dfs",
        help="Names of the train h5 files, should be output of prepare_ground_truth",
        type=str,
        nargs="+",
        required=True,
    )

    ap_var.add_argument("--test_dfs", help="Names of the test h5 files", type=str, nargs="+", required=True)

    ap_var.add_argument(
        "--output_file_prefix", help="Output .pkl file with models, .h5 file with results", type=str, required=True
    )

    ap_var.add_argument(
        "--gt_type",
        help='GT type - "exact" or "approximate"',
        type=GtType,
        choices=list(GtType),
        default=GtType.EXACT,
    )

    ap_var.add_argument(
        "--vcf_type",
        help='VCF type - "single_sample"(GATK) or "deep_variant"',
        type=VcfType,
        choices=list(VcfType),
        default=VcfType.SINGLE_SAMPLE,
    )
    ap_var.add_argument(
        "--custom_annotations",
        help="Custom INFO annotations in the training VCF (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
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
    "Train filtering model"
    np.random.seed(1984)
    random.seed(1984)
    args = parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    logger.debug(args)
    logger.info("Training pipeline: START")
    args.custom_annotations = (
        [x.lower() for x in args.custom_annotations] if args.custom_annotations is not None else []
    )
    try:
        features_to_extract = transformers.get_needed_features(args.vcf_type, args.custom_annotations) + ["label"]
        logger.debug(f"(len(features_to_extract)={len(features_to_extract)}")
        # read all data besides concordance and input_args or as defined in list_of_contigs_to_read
        dfs = []
        for input_file in args.train_dfs:
            df = read_hdf(input_file, columns_subset=features_to_extract)
            dfs.append(df)
        train_df = pd.concat(dfs)
        logger.info("Read training data: success")

        # Train the model
        logger.info("Model training: start")
        model, transformer = variant_filtering_utils.train_model(
            train_df, gt_type=args.gt_type, vtype=args.vcf_type, annots=args.custom_annotations
        )
        logger.info("Model training: done")
        logger.info("Read test data: start")
        dfs = []
        for input_file in args.test_dfs:
            df = read_hdf(input_file, columns_subset=features_to_extract)
            dfs.append(df)
        test_df = pd.concat(dfs)
        logger.info("Read test data: done")

        logger.info("Evaluate training: start")
        train_results = variant_filtering_utils.eval_model(train_df, model, transformer)
        logger.info("Evaluate training: done")
        logger.info("Evaluate test: start")
        test_results = variant_filtering_utils.eval_model(test_df, model, transformer)
        logger.info("Evaluate test: done")
        results_dict = {}
        results_dict["transformer"] = transformer
        results_dict["xgb"] = model
        results_dict["xgb_recall_precision"] = test_results[0]
        results_dict["xgb_recall_precision_curve"] = test_results[1]
        results_dict["xgb_train_recall_precision"] = train_results[0]
        results_dict["xgb_train_recall_precision_curve"] = train_results[1]
        with open(args.output_file_prefix + ".pkl", "wb") as file:
            pickle.dump(results_dict, file)

        accuracy_dfs = []
        prcdict = {}
        for m_var in ("xgb", "xgb_train"):
            name_optimum = f"{m_var}_recall_precision"
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

        logger.info("Model training run: success")

    except Exception as err:
        logger.exception(err)
        logger.error("Model training run: failed")
        raise err


if __name__ == "__main__":
    run(sys.argv)
