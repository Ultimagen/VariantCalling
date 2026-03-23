from __future__ import annotations

import argparse
import json
import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.comparison.quick_fingerprinter import QuickFingerprinter
from ugbio_cloud_utils.cloud_sync import optional_cloud_sync


def __get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quick fingerprinting (finding sample identity of crams, " "given a known list of ground-truth files)",
        description=run.__doc__,
    )
    parser.add_argument(
        "--json_conf",
        required=True,
        help="json file with sample-names, crams, and ground truth files see 'quick_fingerprinting_example.json'",
    )
    parser.add_argument(
        "--region_str",
        type=str,
        default="chr15:26000000-26200000",
        help="region subset string, compare variants only within region",
    )
    parser.add_argument(
        "--add_aws_auth_command", 
        action="store_true", 
        help="add aws auth command to samtools commands"
    )
    parser.add_argument(
        "--regions_bed",
        type=str,
        default=None,
        help="BED file of regions to intersect ground truth and calls with (e.g. exome capture regions). "
             "Results will refer to the intersection of these regions with region_str. Give an entire chromosome for WES data",
    )
    
    parser.add_argument(
        "--min_af_snps",
        type=float,
        default=0.03,
        help="min allele frequency to count as a ground-truth hit",
    )
    parser.add_argument(
        "--min_af_germline_snps",
        type=float,
        default=0.1,
        help="min allele frequency to count a snp as germline snp, for normal-in-tumor <-> normal matching",
    )
    parser.add_argument(
        "--min_hit_fraction_target",
        type=float,
        default=0.99,
        help="fraction of ground-truth variants which has hits in target samples",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="fingerprint",
        help="prefix for output plot PNG filenames",
    )
    return parser


def run(argv):
    """quick fingerprinting to identify known samples in crams"""
    parser = __get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])

    with open(args.json_conf, encoding="utf-8") as fh:
        conf = json.load(fh)

    ref = optional_cloud_sync(conf["references"]["ref_fasta"], args.out_dir)
    optional_cloud_sync(conf["references"]["ref_dict"], args.out_dir)
    optional_cloud_sync(conf["references"]["ref_fasta_index"], args.out_dir)
    cram_files_list = conf["cram_files"]
    ground_truth_vcf_files = conf["ground_truth_vcf_files"]  # dict sample-id -> bed
    hcr_files = conf["ground_truth_hcr_files"]  # dict sample-id -> bed

    region = args.region_str
    regions_bed = args.regions_bed
    min_af_snps = args.min_af_snps
    min_af_germline_snps = args.min_af_germline_snps
    min_hit_fraction_target = args.min_hit_fraction_target

    sp = SimplePipeline(args.fc, args.lc, debug=args.d)
    os.makedirs(args.out_dir, exist_ok=True)
    errors = []

    QuickFingerprinter(
        cram_files_list,
        ground_truth_vcf_files,
        hcr_files,
        ref,
        region,
        min_af_snps,
        min_af_germline_snps,
        min_hit_fraction_target,
        args.add_aws_auth_command,
        args.out_dir,
        sp,
        regions_bed=regions_bed,
        output_prefix=args.output_prefix,
    ).check()

    if len(errors) > 0:
        raise RuntimeError("\n".join(errors))


if __name__ == "__main__":
    import sys

    run(sys.argv)
