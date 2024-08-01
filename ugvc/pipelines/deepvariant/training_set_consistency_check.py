from __future__ import annotations

import argparse
import json
import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.utils.cloud_sync import cloud_sync
from ugvc.validation.variant_hit_fraction_caller import VariantHitFractionCaller


def __get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training_set_consistency_check", description=run.__doc__)
    parser.add_argument("--training_json_conf", required=True, help="json file with training configuration")
    parser.add_argument(
        "--region_str",
        type=str,
        default="chr15:26000000-30000000",
        help="region subset string, compare variants only in this region",
    )
    VariantHitFractionCaller.add_args_to_parser(parser)
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    return parser


def read_count(in_file: str) -> int:
    with open(in_file, encoding="utf-8") as f:
        return int(f.read().strip())


# pylint: disable=too-many-instance-attributes
class TrainingSetConsistency:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        target_crams: list[str],
        normal_crams: list[str],
        ground_truth_vcf: str,
        hcr: str,
        training_intervals_file: str,
        ref: str,
        max_vars: int,
        region: str,
        min_af_snps: int,
        min_af_germline_snps: int,
        min_hit_fraction_target: float,
        out_dir: str,
        sp: SimplePipeline,
    ):
        self.target_crams = target_crams
        self.normal_crams = normal_crams
        self.ground_truth_vcf = ground_truth_vcf
        self.hcr = hcr
        self.training_intervals_file = training_intervals_file
        self.ref = ref
        self.max_vars = max_vars
        self.region = region
        self.out_dir = out_dir
        self.min_af_snps = min_af_snps
        self.min_af_germline_snps = min_af_germline_snps
        self.min_hit_fraction_target = min_hit_fraction_target
        self.sp = sp
        os.makedirs(out_dir, exist_ok=True)
        self.vc = VariantHitFractionCaller(self.ref, self.out_dir, self.sp, self.min_af_snps)

    def check(self):
        errors = []
        suspected_normal_in_tumor_crams = []
        ground_truth_in_hcr = f"{self.out_dir}/ground_truth_in_hcr.vcf.gz"
        ground_truth_in_hcr_and_ti = f"{self.out_dir}/ground_truth_in_hcr_and_ti.vcf.gz"
        ground_truth_to_check = f"{self.out_dir}/ground_truth_to_check.bed"
        ground_truth_to_check_vcf = f"{self.out_dir}/ground_truth_to_check.vcf.gz"
        print("Check consistency:")
        print("    target_crams = \n\t" + "\n\t".join(self.target_crams))
        print("    normal_crams = \n\t" + "\n\t".join(self.normal_crams))
        print(f"    ground_truth_vcf = {self.ground_truth_vcf}")
        print(f"    hcr = {self.hcr}")
        print(f"    training_intervals = {self.training_intervals_file}")

        self.sp.print_and_run(
            f"bedtools intersect -a {self.ground_truth_vcf} -b {self.hcr} -header > {ground_truth_in_hcr}"
        )
        self.sp.print_and_run(
            f"picard IntervalListToBed -I {self.training_intervals_file} -O {self.out_dir}/training_intervals.bed"
        )
        self.sp.print_and_run(
            f"bedtools intersect -a {ground_truth_in_hcr} "
            f"-b {self.out_dir}/training_intervals.bed -header "
            f"| bcftools view -Oz -o {ground_truth_in_hcr_and_ti}"
        )
        self.sp.print_and_run(f"bcftools index -t  {ground_truth_in_hcr_and_ti}")
        self.sp.print_and_run(
            f"bcftools view {ground_truth_in_hcr_and_ti} -r {self.region} -Oz -o {ground_truth_to_check_vcf}"
        )
        self.vc.vcf_to_bed(f"{ground_truth_to_check_vcf}", ground_truth_to_check, max_vars=self.max_vars)
        hcr_intersect_ti = f"{self.out_dir}/hcr_intersected_training_intervals.bed"
        hcr_in_region = f"{self.out_dir}/hcr_in_region.bed"
        self.sp.print_and_run(
            f"bedtools intersect -a {self.hcr} -b {self.out_dir}/training_intervals.bed | "
            f"sort -k 1,1 -k 2,2n > {hcr_intersect_ti}"
        )
        if self.region != "":
            self.sp.print_and_run(f"echo {self.region} | sed 's/:/\t/' | sed 's/-/\t/' > {self.out_dir}/region.bed")
            self.sp.print_and_run(
                f"bedtools intersect -a {hcr_intersect_ti} -b {self.out_dir}/region.bed | "
                f"sort -k 1,1 -k 2,2n > {hcr_in_region}"
            )

        else:
            self.sp.print_and_run(f"cp {hcr_intersect_ti} {hcr_in_region}")

        # Validate that each target cram correlates or anti-correlates the ground-truth
        for target_cram in self.target_crams:
            hit_fraction, hit_count, _ = self.vc.calc_hit_fraction(target_cram, ground_truth_to_check)
            if hit_fraction < self.min_hit_fraction_target:
                if self.normal_crams is None:
                    errors.append(
                        f"{target_cram} - target sample does not match ground truth"
                        f"hit_fraction={hit_fraction}, hit_count={hit_count}"
                    )
                elif hit_fraction > 1 - self.min_hit_fraction_target:
                    errors.append(
                        f"{target_cram} - target sample does not match ground truth"
                        f", and is also not complementary to it, hit_fraction={hit_fraction}, count={hit_count}"
                    )
                else:
                    print(f"{target_cram} - target sample can be normal-in-tumor sample, hit_fraction={hit_fraction}")
                    suspected_normal_in_tumor_crams.append(target_cram)
            else:
                print(f"{target_cram} - target sample match ground truth hit_fraction={hit_fraction}")

        # Validate that each normal cram ani-correlates the ground truth
        normal_germline_vcf_beds = []
        for normal_cram in self.normal_crams:
            hit_fraction, _, _ = self.vc.calc_hit_fraction(normal_cram, ground_truth_to_check)
            if hit_fraction > 1 - self.min_hit_fraction_target:
                errors.append(
                    f"{normal_cram} - normal sample is not complementary to ground truth, hit_fraction={hit_fraction}"
                )
            else:
                print(f"{normal_cram} - normal sample is complementary to ground truth, hit_fraction={hit_fraction}")

            # call germline variants from normal sample, to identify normal-in-tumor samples later
            normal_base_name = f"{os.path.splitext(os.path.basename(normal_cram))[0]}"
            normal_germline_vcf = f"{self.out_dir}/{normal_base_name}.germline.vcf.gz"
            normal_germline_bed = f"{self.out_dir}/{normal_base_name}.germline.bed"
            self.vc.call_variants(normal_cram, normal_germline_vcf, hcr_in_region, min_af=self.min_af_germline_snps)
            self.vc.vcf_to_bed(normal_germline_vcf, normal_germline_bed)
            normal_germline_vcf_beds.append(normal_germline_bed)

        # Validate that each normal-in-tumor sample matches at least one normal sample
        if len(self.normal_crams) > 0:
            for suspected_normal_in_tumor_cram in suspected_normal_in_tumor_crams:
                max_hit_fraction = 0
                best_match = ""
                for normal_germline_bed in normal_germline_vcf_beds:
                    hit_fraction, _, _ = self.vc.calc_hit_fraction(suspected_normal_in_tumor_cram, normal_germline_bed)
                    max_hit_fraction = max(max_hit_fraction, hit_fraction)
                    best_match = normal_germline_bed
                if max_hit_fraction < self.min_hit_fraction_target:
                    errors.append(
                        f"{suspected_normal_in_tumor_cram} - suspected normal-in-tumor sample does "
                        f"not match any normal sample max_hit_fraction={max_hit_fraction}"
                    )
                else:
                    print(
                        f"{suspected_normal_in_tumor_cram} - "
                        f"suspected normal-in-tumor sample matches {best_match} with hit_fraction={max_hit_fraction}"
                    )
        for error in errors:
            print(f"ERROR: {error}")
        return errors


def run(argv):
    """Training set consistency check pipeline."""
    parser = __get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])

    with open(args.training_json_conf, encoding="utf-8") as fh:
        conf = json.load(fh)
    key = list(conf.keys())[0]
    workflow_id = key.split(".")[0]

    ref = cloud_sync(conf[f"{workflow_id}.references"]["ref_fasta"], args.out_dir)
    cloud_sync(conf[f"{workflow_id}.references"]["ref_dict"], args.out_dir)
    cloud_sync(conf[f"{workflow_id}.references"]["ref_fasta_index"], args.out_dir)
    bam_files = conf[f"{workflow_id}.cram_files"]
    background_bam_files = conf[f"{workflow_id}.background_cram_files"]
    ground_truth_vcf_files = conf[f"{workflow_id}.ground_truth_vcf_files"]
    training_hcr_files = conf[f"{workflow_id}.training_hcr_files"]
    training_intervals_files = conf[f"{workflow_id}.training_intervals"]

    region = args.region_str
    min_af_snps = args.min_af_snps
    min_af_germline_snps = args.min_af_germline_snps
    min_hit_fraction_target = args.min_hit_fraction_target

    sp = SimplePipeline(args.fc, args.lc, debug=args.d)
    os.makedirs(args.out_dir, exist_ok=True)
    errors = []

    for i, target_crams in enumerate(bam_files):
        if len(background_bam_files) == len(bam_files):
            normal_crams = background_bam_files[i]
        elif len(background_bam_files) > 0:
            raise RuntimeError("Number of background bam files does not match number of bam files")
        else:
            normal_crams = None

        ground_truth_vcf_file = cloud_sync(ground_truth_vcf_files[i], args.out_dir)
        training_hcr_file = cloud_sync(training_hcr_files[i], args.out_dir)
        training_intervals_file = cloud_sync(training_intervals_files[i], args.out_dir)

        print(f"subset {i}")
        errors.extend(
            TrainingSetConsistency(
                target_crams,
                normal_crams,
                ground_truth_vcf_file,
                training_hcr_file,
                training_intervals_file,
                ref,
                args.max_vars,
                region,
                min_af_snps,
                min_af_germline_snps,
                min_hit_fraction_target,
                f"{args.out_dir}/subset_{i}",
                sp,
            ).check()
        )

    if len(errors) > 0:
        raise RuntimeError("\n".join(errors))
