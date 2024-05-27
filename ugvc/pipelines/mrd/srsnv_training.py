#!/env/python
# Copyright 2023 Ultima Genomics Inc.
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
#    Run single read SNV quality recalibration training
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from simppl.simple_pipeline import SimplePipeline

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.srsnv_plotting_utils import srsnv_report
from ugvc.mrd.srsnv_training_utils import SRSNVTrain


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="srsnv_training.py", description=run.__doc__)
    parser.add_argument(
        "--hom_snv_featuremap",
        type=str,
        required=True,
        help="""input featuremap with homozygote SNVs (True Positive)""",
    )
    parser.add_argument(
        "--single_substitution_featuremap",
        type=str,
        required=True,
        help="""single substitution featuremap (False Positive)""",
    )
    parser.add_argument(
        "--hom_snv_regions",
        type=str,
        required=False,
        help="""Path to bed file containint regions for hom snv (TP) featuremap""",
    )
    parser.add_argument(
        "--single_sub_regions",
        type=str,
        required=True,
        help="""Path to bed file containint regions for single substitution featuremap (FP)""",
    )
    parser.add_argument(
        "--cram_stats_file",
        type=str,
        required=True,
        help="""Path to cram stats file (for LoD estimation)""",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        required=False,
        help="""Path to json file with input parameters for the classification model""",
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        help="""Size of the train set for the classification model""",
    )
    parser.add_argument(
        "--test_set_size",
        type=int,
        default=None, 
        help="""Size of the test set for the classification model""",
    )
    parser.add_argument(
        "--num_CV_folds", 
        type=int,
        default=1, 
        help="""Number of cross-validation folds to use. Default=1 (no CV)"""
    )
    parser.add_argument(
        "--split_folds_randomly", 
        action="store_true",
        help="""by default the training data is split into folds by chromosomes, if the flag is provided it is split randomly."""
    )
    parser.add_argument(
        "--numerical_features",
        type=str,
        nargs="+",
        help="""comma separated list of numerical features for ML classifier """,
    )
    parser.add_argument(
        "--categorical_features",
        type=str,
        nargs="+",
        help="""comma separated list of categorical features for ML classifier """,
    )
    parser.add_argument(
        "--balanced_sampling_info_fields",
        type=str,
        nargs="+",
        default=None,
        help="comma separated list of categorical features to be used for balanced sampling of the TP training set"
        " to eliminate prior distribution bias (e.g. 'trinuc_context_with_alt,is_forward')",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Path to which output files will be written to""",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="",
        required=False,
        help="""basename of output files""",
    )
    parser.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="""reference fasta, only required for motif annotation""",
    )
    parser.add_argument(
        "--flow_order",
        type=str,
        required=False,
        default=DEFAULT_FLOW_ORDER,
        help="""flow order - required for cycle skip annotation """,
    )
    parser.add_argument(
        "--lod_filters",
        type=str,
        required=False,
        default=None,
        help="""json file with a dict of format 'filter name':'query' for LoD simulation """,
    )
    parser.add_argument(
        "--balanced_strand_adapter_version",
        type=str,
        required=False,
        default=None,
        help="""adapter version, indicates if input featuremap is from balanced ePCR data """,
    )
    parser.add_argument(
        "--pre_filter",
        type=str,
        required=False,
        default=None,
        help="""bcftools include filter to apply as part of a "bcftools view <vcf> -i 'pre_filter' command""",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default=42,
        help="""random seed for reproducibility""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Train a model for single read SNV quality recalibration"""
    args = parse_args(argv)
    simple_pipeline_args = (0, 10000, False)
    sp = SimplePipeline(
        simple_pipeline_args[0],
        simple_pipeline_args[1],
        debug=simple_pipeline_args[2],
        print_timing=True,
    )

    # TODO add the option to read from a json file         model_parameters: dict | str = None,
    # TODO add to args         classifier_class=xgb.XGBClassifier,

    s = SRSNVTrain(
        tp_featuremap=args.hom_snv_featuremap,
        fp_featuremap=args.single_substitution_featuremap,
        tp_regions_bed_file=args.hom_snv_regions,
        fp_regions_bed_file=args.single_sub_regions,
        numerical_features=args.numerical_features,
        categorical_features=args.categorical_features,
        balanced_sampling_info_fields=args.balanced_sampling_info_fields
        if args.balanced_sampling_info_fields
        else None,
        sorter_json_stats_file=args.cram_stats_file,
        train_set_size=args.train_set_size,
        test_set_size=args.test_set_size,
        k_folds=args.num_CV_folds, 
        split_folds_by_chrom=not args.split_folds_randomly, 
        out_path=args.output,
        out_basename=args.basename,
        lod_filters=args.lod_filters,
        balanced_strand_adapter_version=args.balanced_strand_adapter_version,
        pre_filter=args.pre_filter,
        random_seed=args.random_seed,
        simple_pipeline=sp,
    ).process()

    # TODO: merge the two reports so train and test set results are presented together
    srsnv_report(
        out_path=args.output,
        out_basename=args.basename,
        report_name="test",
        model_file=s.model_save_path,
        params_file=s.params_save_path,
        simple_pipeline=None,
    )

    srsnv_report(
        out_path=args.output,
        out_basename=args.basename,
        report_name="train",
        model_file=s.model_save_path,
        params_file=s.params_save_path,
        simple_pipeline=None,
    )
