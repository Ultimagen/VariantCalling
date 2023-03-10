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

from ugvc.mrd.mrd_utils import featuremap_to_dataframe


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="featuremap_to_dataframe", description=run.__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input featuremap file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output dataframe will be written, if None a file with the same name as input and
".parquet" extension will be created""",
    )
    parser.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="""reference fasta, only required for motif annotation
most likely gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta but it must be localized""",
    )
    parser.add_argument(
        "-f",
        "--flow_order",
        type=str,
        required=False,
        default=None,
        help="""flow order - required for cycle skip annotation but not mandatory""",
    )
    parser.add_argument(
        "-m",
        "--motif_length",
        type=int,
        default=4,
        help="motif length to annotate the vcf with",
    )
    parser.add_argument(
        "--report_sense_strand_bases",
        default=False,
        action="store_true",
        help="if True, the ref, alt, and motifs will be reported according to the sense strand "
        "and not according to the read orientation",
    )
    parser.add_argument(
        "--show_progress_bar",
        default=False,
        action="store_true",
        help="show progress bar (tqdm)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--matched", action="store_true")
    group.add_argument("--control", action="store_true")
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Convert featuremap to pandas dataframe"""
    args_in = __parse_args(argv)
    is_matched = None if args_in.matched is None else bool(args_in.matched)

    featuremap_to_dataframe(
        featuremap_vcf=args_in.input,
        output_file=args_in.output,
        reference_fasta=args_in.reference_fasta,
        motif_length=args_in.motif_length,
        report_read_strand=not args_in.report_sense_strand_bases,
        show_progress_bar=args_in.show_progress_bar,
        flow_order=args_in.flow_order,
        is_matched=is_matched,
    )
    print("DONE")
