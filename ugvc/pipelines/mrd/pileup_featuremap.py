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

import numpy as np

from ugvc.mrd.featuremap_consensus_utils import pileup_featuremap, pileup_featuremap_on_an_interval_list


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pileup_featuremap", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--featuremap",
        type=str,
        required=True,
        help="""Featuremap vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output_vcf",
        type=str,
        required=True,
        help="""Output pileup vcf file""",
    )
    interval_group = parser.add_mutually_exclusive_group()
    interval_group.add_argument(
        "-i",
        "--genomic_interval",
        type=str,
        required=False,
        default=None,
        help="""Genomic interval to pileup, format: chr:start-end (default: None).\
            Mutually exclusive with --interval_list""",
    )
    interval_group.add_argument(
        "-il",
        "--interval_list",
        type=str,
        required=False,
        default=None,
        help="""Interval list file (default: None).\
            Mutually exclusive with --genomic_interval""",
    )
    parser.add_argument(
        "-q",
        "--min_qual",
        type=int,
        required=False,
        default=0,
        help="""Quality filter threshold (default: 0)""",
    )
    parser.add_argument(
        "-s",
        "--sample_name",
        type=str,
        required=False,
        default="SAMPLE",
        help="""Sample name (default: SAMPLE)""",
    )
    parser.add_argument(
        "-qf",
        "--qual_agg_func",
        type=callable,
        required=False,
        default=np.max,
        help="""Function to aggregate quality scores (default: np.max)""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Collapses a featuremap vcf file into a pileup vcf file"""
    args_in = __parse_args(argv)
    if args_in.genomic_interval is None and args_in.interval_list is None:
        # no genomic interval or interval list provided
        # run pileup_featuremap on the whole file
        pileup_featuremap(
            featuremap=args_in.featuremap,
            output_vcf=args_in.output_vcf,
            genomic_interval=args_in.genomic_interval,
            min_qual=args_in.min_qual,
            sample_name=args_in.sample_name,
            qual_agg_func=args_in.qual_agg_func,
        )
    elif args_in.genomic_interval is not None and args_in.interval_list is None:
        # only genomic_interval provided
        # run pileup_featuremap on the genomic_interval
        pileup_featuremap(
            featuremap=args_in.featuremap,
            output_vcf=args_in.output_vcf,
            genomic_interval=args_in.genomic_interval,
            min_qual=args_in.min_qual,
            sample_name=args_in.sample_name,
            qual_agg_func=args_in.qual_agg_func,
        )
    elif args_in.genomic_interval is None and args_in.interval_list is not None:
        # only interval_list provided
        # run pileup_featuremap on the interval list
        pileup_featuremap_on_an_interval_list(
            featuremap=args_in.featuremap,
            output_vcf=args_in.output_vcf,
            interval_list=args_in.interval_list,
            min_qual=args_in.min_qual,
            sample_name=args_in.sample_name,
            qual_agg_func=args_in.qual_agg_func,
        )
