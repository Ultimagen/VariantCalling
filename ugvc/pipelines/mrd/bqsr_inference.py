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

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.bqsr_inference_utils import bqsr_inference


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bqsr_inference.py", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--featuremap_path",
        type=str,
        required=True,
        help="""input featuremap file""",
    )
    parser.add_argument(
        "-f",
        "--params_path",
        type=str,
        required=True,
        help="""params file path""",
    )
    parser.add_argument(
        "-f",
        "--model_path",
        type=str,
        required=True,
        help="""xgb model file path""",
    )
    parser.add_argument(
        "-o",
        "--output_path",
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
        help="""reference fasta, only required for motif annotation most likely"""
    )
    parser.add_argument(
        "--flow_order",
        type=str,
        required=False,
        default=DEFAULT_FLOW_ORDER,
        help="""flow order - required for cycle skip annotation """,
    )
    parser.add_argument(
        "--chrom",
        type=str,
        required=False,
        default=None,
        help="""training chromosome""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """BQSR inference: load model, run inference and write quality for all reads in the input featuremap"""
    args = parse_args(argv)

    bqsr_inference(
        featuremap_path=args.featuremap_path,
        params_path=args.params_path,
        model_path=args.model_path,
        out_path=args.output_path,
        out_basename=args.basename,
    )
