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
from ugvc.mrd.srsnv_inference_utils import single_read_snv_inference


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
    
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """BQSR inference: load model, run inference and write quality for all events in the input featuremap"""
    args = parse_args(argv)

    single_read_snv_inference(
        featuremap_path=args.featuremap_path,
        params_path=args.params_path,
        model_path=args.model_path,
        out_path=args.output_path,
    )

if __name__ == '__main__':
    import sys
    run(sys.argv)