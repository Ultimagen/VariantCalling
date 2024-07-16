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
#    Add additional feature annotations to featuremap, to be used from single-read SNV qual recalibration
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.mrd_utils import annotate_featuremap
from ugvc.mrd.ppmSeq_utils import supported_adapter_versions


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="annotate featuremap", description=run.__doc__)
    parser.add_argument(
        "-i",
        "--featuremap_path",
        type=str,
        required=True,
        help="input featuremap file",
    )
    parser.add_argument(
        "--ppmSeq_adapter_version",
        choices=supported_adapter_versions,
        required=False,
        help="ppmSeq adapter version",
    )
    parser.add_argument(
        "-o",
        "--output_featuremap",
        type=str,
        required=True,
        help="Path of annotated featuremap file",
    )
    parser.add_argument("-r", "--ref_fasta", type=str, required=True, help="Reference genome fasta file")
    parser.add_argument(
        "--flow_order",
        type=str,
        default=DEFAULT_FLOW_ORDER,
    )
    parser.add_argument(
        "--motif_length_to_annotate",
        type=int,
        default=3,
    )
    parser.add_argument("--max_hmer_length", type=int, default=20)

    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Add additional feature annotations to featuremap"""
    args = parse_args(argv)

    annotate_featuremap(
        input_featuremap=args.featuremap_path,
        output_featuremap=args.output_featuremap,
        ref_fasta=args.ref_fasta,
        ppmSeq_adapter_version=args.ppmSeq_adapter_version,
        flow_order=args.flow_order,
        motif_length_to_annotate=args.motif_length_to_annotate,
        max_hmer_length=args.max_hmer_length,
    )
