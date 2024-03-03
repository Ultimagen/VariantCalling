#!/usr/bin/env python
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
#    Prepare data for training

from __future__ import annotations

import argparse
import logging
import sys

from ugvc.filtering import tprep_constants, training_prep


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(prog="training_prep_pipeline.py", description="Prepare training data")
    ap_var.add_argument("--call_vcf", help="Call VCF file", type=str, required=True)
    ap_var.add_argument(
        "--gt_type",
        help="Ground truth type: exact or approximate",
        type=tprep_constants.GtType,
        required=True,
        default=tprep_constants.GtType.EXACT,
    )
    ap_var.add_argument("--blacklist", help="Blacklist file", type=str, required=False)
    ap_var.add_argument(
        "--base_vcf", help="Truth VCF file, needed only if gt_type is 'exact'", type=str, required=False
    )
    ap_var.add_argument(
        "--reference", help="Reference file prefix. Requires .fai,and .sdf folder", type=str, required=False
    )
    ap_var.add_argument("--hcr", help="High confidence regions BED file", type=str, required=False)
    ap_var.add_argument(
        "--contigs_to_read",
        help="List of chromosomes to read the data from, default: all",
        nargs="+",
        type=str,
        required=False,
    )
    ap_var.add_argument(
        "--contig_for_test",
        help="Chromosome to split into test set, contained in contigs_to_read",
        type=str,
        required=False,
    )
    ap_var.add_argument("--output_prefix", help="Output HDF5 files prefix", type=str, required=True)
    return ap_var.parse_args(argv)


def run(argv: list[str]):
    """Run function"""
    args = parse_args(argv)
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Prepare training data started")
    if args.gt_type == tprep_constants.GtType.EXACT:
        assert args.blacklist is None
        assert args.base_vcf is not None
        assert args.reference is not None
        assert args.hcr is not None
        training_prep.prepare_ground_truth(
            args.call_vcf,
            args.base_vcf,
            args.hcr,
            args.reference,
            args.output_prefix + ".h5",
            args.contigs_to_read,
            args.contig_for_test,
        )
    elif args.gt_type == tprep_constants.GtType.APPROXIMATE:
        assert args.blacklist is not None
        assert args.base_vcf is None
        assert args.reference is None
        assert args.hcr is None
        training_prep.label_with_approximate_gt(
            args.call_vcf, args.blacklist, args.output_prefix + ".h5", args.contigs_to_read, args.contig_for_test
        )
    logger.info("Prepare training data finished")
    return 0


if __name__ == "__main__":
    run(sys.argv[1:])
