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
#    Calculate error rate per motif
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from simppl.simple_pipeline import SimplePipeline

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.bqsr_train_utils import BQSRTrain, bqsr_train_report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bqsr_train.py", description=run.__doc__)
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
        required=False,
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
    return parser.parse_args(argv[1:])


def run(argv: list[str]):

    args = parse_args(argv)
    simple_pipeline_args = (0, 10000, False)
    sp = SimplePipeline(
        simple_pipeline_args[0],
        simple_pipeline_args[1],
        debug=simple_pipeline_args[2],
        print_timing=True,
    )

    bqsrtrain = BQSRTrain(
        tp_featuremap=args.hom_snv_featuremap,
        fp_featuremap=args.single_substitution_featuremap,
        tp_regions_bed_file=args.hom_snv_regions,
        fp_regions_bed_file=args.single_sub_regions,
        cram_stats_file=args.cram_stats_file,
        out_path=args.output,
        out_basename=args.basename,
        reference_fasta=args.reference_fasta,
        flow_order=args.flow_order,
        model_params_file=args.model_params,
        simple_pipeline=sp,
    ).process()

    return
    bqsr_train_report(
        out_path=bqsrtrain.workdir,
        out_basename=bqsrtrain.out_basename,
        report_name="test",
        model_path=bqsrtrain.model_save_path,
        X_path=bqsrtrain.X_test_save_path,
        y_path=bqsrtrain.y_test_save_path,
        fp_bed_file=args.single_sub_regions,
        params_path=bqsrtrain.params_save_path,
        simple_pipeline=sp,
    )

    bqsr_train_report(
        out_path=bqsrtrain.workdir,
        out_basename=bqsrtrain.out_basename,
        report_name="train",
        model_path=bqsrtrain.model_save_path,
        X_path=bqsrtrain.X_train_save_path,
        y_path=bqsrtrain.y_train_save_path,
        fp_bed_file=args.single_sub_regions,
        params_path=bqsrtrain.params_save_path,
        simple_pipeline=sp,
    )