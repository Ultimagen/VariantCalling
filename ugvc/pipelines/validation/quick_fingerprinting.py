from __future__ import annotations

import argparse
import json
import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.utils.cloud_sync import cloud_sync
from ugvc.validation.variant_hit_fraction_caller import VariantHitFractionCaller


def __get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training_set_consistency_check", description=run.__doc__)
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


# pylint: disable=too-many-instance-attributes
class QuickFingerprinting:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        sample_crams: dict[list[str]],
        ground_truth_vcfs: dict[str, str],
        hcrs: dict[str, str],
        ref: str,
        max_vars: int,
        region: str,
        min_af_snps: int,
        min_af_germline_snps: int,
        min_hit_fraction_target: float,
        out_dir: str,
        sp: SimplePipeline,
    ):
        self.crams = sample_crams
        self.ground_truth_vcfs = ground_truth_vcfs
        self.hcrs = hcrs
        self.ref = ref
        self.max_vars = max_vars
        self.region = region
        self.out_dir = out_dir
        self.min_af_snps = min_af_snps
        self.min_af_germline_snps = min_af_germline_snps
        self.min_hit_fraction_target = min_hit_fraction_target
        self.sp = sp
        self.vc = VariantHitFractionCaller(self.ref, self.out_dir, self.sp, self.min_af_snps)

        os.makedirs(out_dir, exist_ok=True)

        self.ground_truth_to_check_beds = self.prepare_ground_truth()

    def prepare_ground_truth(self):
        ground_truth_to_check_beds = {}
        self.sp.print_and_run(f"echo {self.region} | sed 's/:/\t/' | sed 's/-/\t/' > {self.out_dir}/region.bed")

        for sample_id in self.ground_truth_vcfs:
            ground_truth_vcf = cloud_sync(self.ground_truth_vcfs[sample_id], self.out_dir)
            hcr = cloud_sync(self.hcrs[sample_id], self.out_dir)
            ground_truth_in_hcr = f"{self.out_dir}/{sample_id}_ground_truth_in_hcr.vcf.gz"
            ground_truth_to_check_vcf = f"{self.out_dir}/{sample_id}_ground_truth_to_check.vcf.gz"
            ground_truth_to_check_bed = f"{self.out_dir}/{sample_id}_ground_truth_to_check.bed"
            hcr_in_region = f"{self.out_dir}/{sample_id}_hcr_in_region.bed"

            self.sp.print_and_run(
                f"bedtools intersect -a {ground_truth_vcf} -b {hcr} -header | "
                + "bcftools view -Oz -o  {ground_truth_in_hcr}"
            )
            self.sp.print_and_run(f"bcftools index -t  {ground_truth_in_hcr}")
            self.sp.print_and_run(
                f"bcftools view {ground_truth_in_hcr} -r {self.region} -Oz -o {ground_truth_to_check_vcf}"
            )
            self.vc.vcf_to_bed(f"{ground_truth_to_check_vcf}", ground_truth_to_check_bed, max_vars=self.max_vars)

            # before we sorted hcr here, do we need to sort?

            if self.region != "":
                self.sp.print_and_run(
                    f"bedtools intersect -a {hcr} -b {self.out_dir}/region.bed | "
                    f"sort -k 1,1 -k 2,2n > {hcr_in_region}"
                )
            else:
                self.sp.print_and_run(f"cp {hcr} {hcr_in_region}")

            ground_truth_to_check_beds[sample_id] = ground_truth_to_check_bed
        return ground_truth_to_check_beds

    def check(self):
        errors = []

        for sample_id in self.crams:
            print(f"Check consistency for {sample_id}:")
            crams = self.crams[sample_id]

            print("    crams = \n\t" + "\n\t".join(self.crams[sample_id]))
            print(f"    hcrs = {self.hcrs}")
            print(f"    ground_truth_vcfs = {self.ground_truth_vcfs}")

            for cram in crams:
                # Validate that each cram correlates to the ground-truth
                print("")
                hit_fractions = []
                max_hit_fraction = 0
                best_match = None
                match_to_expected_truth = None
                cram_base_name = os.path.basename(cram)
                local_cram = f"{self.out_dir}/{cram_base_name}"
                self.sp.print_and_run(f"samtools view {cram} {self.region} -C -o {local_cram}")
                for ground_truth_id, ground_truth_to_check_bed in self.ground_truth_to_check_beds.items():
                    hit_fraction, hit_count, _ = self.vc.calc_hit_fraction(local_cram, ground_truth_to_check_bed)
                    if hit_fraction > max_hit_fraction:
                        max_hit_fraction = hit_fraction
                        best_match = ground_truth_id
                    hit_fractions.append(hit_fraction)
                    if sample_id == ground_truth_id and hit_fraction < self.min_hit_fraction_target:
                        match_to_expected_truth = hit_fraction
                        errors.append(
                            f"{cram} - {sample_id} does not match ground truth"
                            f"hit_fraction={hit_fraction}, hit_count={hit_count}"
                        )
                    elif sample_id != ground_truth_id and hit_fraction > self.min_hit_fraction_target:
                        errors.append(
                            f"{cram} - {sample_id} matched ground truth of {ground_truth_id}"
                            f", hit_fraction={hit_fraction}, count={hit_count}"
                        )

                    print(f"{cram} - {sample_id} vs. {ground_truth_id} hit_fraction={hit_fraction}")
                if best_match != sample_id:
                    if match_to_expected_truth is None:
                        print(f"best_match={best_match} hit_fraction={max_hit_fraction}")
                    else:
                        errors.append(
                            f"{cram} - {sample_id} does not have a maximal match with it's expected ground truth"
                            f"hit_fraction={match_to_expected_truth}, max_hit_fraction = {max(hit_fractions)}"
                        )
        for error in errors:
            print(f"ERROR: {error}")
        return errors


def run(argv):
    """quick fingerprinting to identify known samples in crams"""
    parser = __get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])

    with open(args.json_conf, encoding="utf-8") as fh:
        conf = json.load(fh)

    ref = cloud_sync(conf["references"]["ref_fasta"], args.out_dir)
    cloud_sync(conf["references"]["ref_dict"], args.out_dir)
    cloud_sync(conf["references"]["ref_fasta_index"], args.out_dir)
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

    QuickFingerprinting(
        cram_files_list,
        ground_truth_vcf_files,
        hcr_files,
        ref,
        args.max_vars,
        region,
        min_af_snps,
        min_af_germline_snps,
        min_hit_fraction_target,
        args.out_dir,
        sp,
    ).check()

    if len(errors) > 0:
        raise RuntimeError("\n".join(errors))


if __name__ == "__main__":
    import sys

    run(sys.argv)
