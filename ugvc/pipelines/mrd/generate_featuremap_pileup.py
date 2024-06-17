#!/env/python
# Copyright 2024 Ultima Genomics Inc.
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
#    Generates multiple synthetic signatures from a database, with the same
#    trinucleotide substitution context as the input signature.
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from ugvc.mrd.featuremap_consensus_utils import generate_featuremap_pileup


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_featuremap_pileup", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--featuremap",
        type=str,
        required=True,
        help="""Featuremap vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="""Output directory for the pileup vcf file""",
    )
    parser.add_argument(
        "-q",
        "--min_qual",
        type=int,
        required=False,
        default=40,
        help="""Quality filter threshold""",
    )
    parser.add_argument(
        "--vcf_header",
        type=str,
        required=False,
        default="/data/rare_variants/giab_mixes/featuremap_consensus_test.header.txt",
        help="""Txt file of the vcf header""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Generates multiple synthetic signatures from a database,
    with the same trinucleotide substitution context as the input signature"""
    args_in = __parse_args(argv)
    generate_featuremap_pileup(
        featuremap=args_in.featuremap,
        output_dir=args_in.output_dir,
        min_qual=args_in.min_qual,
        vcf_header=args_in.vcf_header,
    )
