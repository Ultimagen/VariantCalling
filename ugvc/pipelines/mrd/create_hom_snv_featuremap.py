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
#    Create a HOM SNV featuremap from a featuremap
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from simppl.simple_pipeline import SimplePipeline

from ugvc.mrd.featuremap_utils import create_hom_snv_featuremap


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="create_hom_snv_featuremap.py", description=run.__doc__)
    parser.add_argument(
        "--featuremap",
        type=str,
        required=True,
        help="""Input featuremap""",
    )
    parser.add_argument(
        "--sorter_stats_json",
        type=str,
        default=None,
        help="Path to Sorter statistics JSON file. If None, the minimum coverage "
        "will be set to requested_min_coverage even if the median coverage is lower, might yield an empty output.",
    )
    parser.add_argument(
        "--hom_snv_featuremap",
        type=str,
        default=None,
        help="Output featuremap with HOM SNVs reads to be used as True Positives. If None (default), "
        "the hom_snv_featuremap will be the same as the input featuremap with a '.hom_snv.vcf.gz' suffix.",
    )
    parser.add_argument(
        "--requested_min_coverage",
        type=int,
        default=20,
        help="Minimum coverage requested for locus to be propagated to the output. If the median coverage "
        "is lower than this value, the median coverage will be used as the minimum coverage instead.",
    )
    parser.add_argument(
        "--min_af",
        type=float,
        default=0.7,
        help="Minimum allele fraction in the featuremap to be considered a HOM SNV"
        "The default is chosen as 0.7 and not higher because some SNVs are pre-filtered from the FeatureMap due to "
        "MAPQ<60 or due to adjacent hmers.",
    )

    return parser


def run(argv: list[str]):
    """Create a HOM SNV featuremap from a featuremap"""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    sp = SimplePipeline(args.fc, args.lc, debug=args.d, print_timing=True)

    create_hom_snv_featuremap(
        featuremap=args.featuremap,
        sorter_stats_json=args.sorter_stats_json,
        hom_snv_featuremap=args.hom_snv_featuremap,
        sp=sp,
        requested_min_coverage=args.requested_min_coverage,
        min_af=args.min_af,
    )
