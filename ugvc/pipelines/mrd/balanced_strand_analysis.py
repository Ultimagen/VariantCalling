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
#    Converts featuremap VCF-like file to dataframe
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from ugvc.mrd.balanced_strand_utils import (
    MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
    MIN_STEM_END_MATCHED_LENGTH,
    MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    STRAND_RATIO_LOWER_THRESH,
    STRAND_RATIO_UPPER_THRESH,
    balanced_strand_analysis,
    supported_adapter_versions,
)


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="featuremap_to_dataframe", description=run.__doc__)
    parser.add_argument(
        "--adapter-version",
        choices=[v.value for v in supported_adapter_versions],
        help="Library adapter version",
    )
    parser.add_argument(
        "--trimmer-histogram-csv",
        type=str,
        help="path to a balanced strand Trimmer histogram file",
    )
    parser.add_argument(
        "--sorter-stats-csv",
        type=str,
        help="path to a Sorter stats file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path (folder) to which data and report will be written to",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default=None,
        help="basename for output files",
    )
    parser.add_argument(
        "--input-material-ng",
        type=float,
        required=False,
        default=None,
        help="Optional - input material in ng, will be included in statistics and report",
    )
    parser.add_argument(
        "--sr-lower",
        type=float,
        default=STRAND_RATIO_LOWER_THRESH,
        help="lower strand ratio threshold for determining strand ratio category",
    )
    parser.add_argument(
        "--sr-upper",
        type=float,
        default=STRAND_RATIO_UPPER_THRESH,
        help="upper strand ratio threshold for determining strand ratio category",
    )
    parser.add_argument(
        "--min-tot-hmer",
        type=int,
        default=MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
        help="minimum total hmer lengths in tags for determining strand ratio category",
    )
    parser.add_argument(
        "--max-tot-hmer",
        type=int,
        default=MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
        help="maximum total hmer lengths in tags for determining strand ratio category",
    )
    parser.add_argument(
        "--min-stem-length",
        type=int,
        default=MIN_STEM_END_MATCHED_LENGTH,
        help="minimum length of stem end matched to determine the read end was reached",
    )
    parser.add_argument(
        "--generate-report",
        type=bool,
        required=False,
        default=True,
        help="""generate an html + jupyter report""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Convert featuremap to pandas dataframe"""
    args_in = __parse_args(argv)

    balanced_strand_analysis(
        adapter_version=args_in.adapter_version,
        trimmer_histogram_csv=args_in.trimmer_histogram_csv,
        sorter_stats_csv=args_in.sorter_stats_csv,
        output_path=args_in.output_path,
        output_basename=args_in.output_basename,
        collect_statistics_kwargs=dict(input_material_ng=args_in.input_material_ng)
        if args_in.input_material_ng
        else None,
        generate_report=args_in.generate_report,
        sr_lower=args_in.sr_lower,
        sr_upper=args_in.sr_upper,
        min_total_hmer_lengths_in_tags=args_in.min_tot_hmer,
        max_total_hmer_lengths_in_tags=args_in.max_tot_hmer,
        min_stem_end_matched_length=args_in.min_stem_length,
    )