from __future__ import annotations

import argparse
import json
import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.comparison.quick_fingerprinter import QuickFingerprinter
from ugvc.comparison.variant_hit_fraction_caller import VariantHitFractionCaller
from ugvc.utils.cloud_sync import optional_cloud_sync


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
        help="region subset string, compare variants only in this region",
    )
    
    VariantHitFractionCaller.add_args_to_parser(parser)
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
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
        args.out_dir,
        sp
    ).check()

    if len(errors) > 0:
        raise RuntimeError("\n".join(errors))


if __name__ == "__main__":
    import sys

    run(sys.argv)
