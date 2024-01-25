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
#    Generates multiple synthetic signatures from a database, with the same
#    trinucleotide substitution context as the input signature.
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

from ugvc.mrd.mrd_utils import generate_synthetic_signatures


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_synthetic_signature_from_db", description=run.__doc__)
    parser.add_argument(
        "-s",
        "--signature_vcf",
        type=str,
        required=True,
        help="""Signature vcf file""",
    )
    parser.add_argument(
        "-db",
        "--db_vcf",
        type=str,
        required=True,
        help="""Database vcf file (for example, PCAWG)""",
    )
    parser.add_argument(
        "-n",
        "--n_synthetic_signatures",
        type=int,
        required=True,
        help="""Number of synthetic signatures to generate""",
    )
    parser.add_argument(
        "-r",
        "--ref_fasta",
        type=str,
        required=False,
        help="reference fasta file, default None. Required if input vcf is not annotated with left and right motifs "
        "X_LM and X_RM",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=False,
        help="""Output directory for synthetic signatures""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Generates multiple synthetic signatures from a database,
    with the same trinucleotide substitution context as the input signature"""
    args_in = __parse_args(argv)
    generate_synthetic_signatures(
        signature_vcf=args_in.signature_vcf,
        db_vcf=args_in.db_vcf,
        n_synthetic_signatures=args_in.n_synthetic_signatures,
        output_dir=args_in.output_dir,
        ref_fasta=args_in.ref_fasta,
    )
