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
#    Profile of errors per motif
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.positional_substitution_error_rate_utils import (
    POSITIONAL_DEFAULT_ERROR_PER_PREFIX,
    POSITIONAL_DEFAULT_FILE_PREFIX,
    POSITIONAL_MAX_POSITIONS_PLOT,
    POSITIONAL_PLOT_STEP_SIZE,
    POSITIONAL_X_EDIST_THRESHOLD,
    POSITIONAL_X_SCORE_THRESHOLD,
    calc_positional_error_rate_profile,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="positional_error_rate_profile.py", description=run.__doc__)

    parser.add_argument(
        "--featuremap_single_substitutions_dataframe",
        type=str,
        required=True,
        help="""featuremap_single_substitutions_dataframe parquet file""",
    )
    parser.add_argument(
        "--coverage_per_motif",
        type=str,
        required=True,
        help="""coverage_per_motif h5 file""",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Path to which output folder the dataframe will be written (multiple files)""",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        required=False,
        default=POSITIONAL_DEFAULT_FILE_PREFIX,
        help="""Prefix of output file names for .png files""",
    )
    parser.add_argument(
        "--error_per_pos_file_prefix",
        type=str,
        required=False,
        default=POSITIONAL_DEFAULT_ERROR_PER_PREFIX,
        help="""Prefix of output file names for .png files""",
    )
    parser.add_argument(
        "--flow_order",
        type=str,
        required=False,
        default=DEFAULT_FLOW_ORDER,
        help="""flow order - required for cycle skip annotation """,
    )
    parser.add_argument(
        "--allow_softlclip",
        type=bool,
        required=False,
        default=False,
        help=""" if True: include reads with softclip in the analysis (default False) """,
    )
    parser.add_argument(
        "--position_plot_step_size",
        type=int,
        required=False,
        default=POSITIONAL_PLOT_STEP_SIZE,
        help=""" base pair resolution of output plot """,
    )
    parser.add_argument(
        "--max_position_for_plot",
        type=int,
        required=False,
        default=POSITIONAL_MAX_POSITIONS_PLOT,
        help=""" max position for plot """,
    )
    parser.add_argument(
        "--edist_threshold",
        type=int,
        required=False,
        default=POSITIONAL_X_EDIST_THRESHOLD,
        help=""" Threshold of edit distance from the reference genome for the inclusion of reads""",
    )
    parser.add_argument(
        "--xscore_threshold",
        type=int,
        required=False,
        default=POSITIONAL_X_SCORE_THRESHOLD,
        help=""" X Score threshold (log likelihood difference between allele containing a substitution and the ref)""",
    )

    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Calculate positional substitution error rate profile for cycle skip motifs"""
    print(f"positional_error_rate_profile.run called with {argv}")
    args = parse_args(argv)
    calc_positional_error_rate_profile(
        single_substitutions_file_name=args.featuremap_single_substitutions_dataframe,
        coverage_per_motif_file_name=args.coverage_per_motif,
        output_folder=args.output,
        output_file_prefix=args.output_file_prefix,
        error_per_pos_file_prefix=args.error_per_pos_file_prefix,
        allow_softlclip=args.allow_softlclip,
        flow_order=args.flow_order,
        position_plot_step_size=args.position_plot_step_size,
        max_positions_plot=args.max_position_for_plot,
        edist_threshold=args.edist_threshold,
        xscore_threshold=args.xscore_threshold,
    )
