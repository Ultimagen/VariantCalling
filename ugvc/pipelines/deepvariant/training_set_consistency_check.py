from __future__ import annotations

import argparse
import json
import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.utils.cloud_sync import cloud_sync


def __get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training_set_consistency_check", description=run.__doc__)
    parser.add_argument("--training_json_conf", required=True, help="json file with training configuration")
    parser.add_argument(
        "--region_str",
        type=str,
        default="chr1:1000000-2000000",
        help="region subset string, compare variants only in this region",
    )
    parser.add_argument("--max_vars", type=int, default=2000, help="max number of variants to check for concordance")
    parser.add_argument(
        "--min_count_snps", type=int, default=3, help="number of snp supporting reads to count as a ground-truth hit"
    )
    parser.add_argument(
        "--min_count_germline_snps",
        type=int,
        default=10,
        help="number of snp supporting reads to count as a germline snp, for normal-in-tumor <-> normal matching",
    )
    parser.add_argument(
        "--min_hit_fraction_target",
        type=float,
        default=0.98,
        help="fraction of ground-truth variants which has hits in target samples",
    )
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
        min_count_snps: int,
        min_count_germline_snps: int,
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
        self.min_count_snps = min_count_snps
        self.min_count_germline_snps = min_count_germline_snps
        self.min_hit_fraction_target = min_hit_fraction_target
        self.sp = sp
        os.makedirs(out_dir, exist_ok=True)

    def check(self):
        errors = []
        suspected_normal_in_tumor_crams = []
        ground_truth_in_hcr = f"{self.out_dir}/ground_truth_in_hcr.vcf"
        ground_truth_in_hcr_and_ti = f"{self.out_dir}/ground_truth_in_hcr_and_ti.vcf"
        ground_truth_to_check = f"{self.out_dir}/ground_truth_to_check.bed"
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
            f"-b {self.out_dir}/training_intervals.bed -header > {ground_truth_in_hcr_and_ti}"
        )
        self.vcf_to_bed(ground_truth_in_hcr_and_ti, ground_truth_to_check, max_vars=self.max_vars)
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

        # Validate that eacg target cram correlates or anti-correlates the ground-truth
        for target_cram in self.target_crams:
            hit_fraction = self.calc_hit_fraction(target_cram, ground_truth_to_check)
            if hit_fraction < self.min_hit_fraction_target:
                if self.normal_crams is None:
                    errors.append(
                        f"{target_cram} - target sample does not match ground truth hit_fraction={hit_fraction}"
                    )
                elif hit_fraction > 1 - self.min_hit_fraction_target:
                    errors.append(
                        f"{target_cram} - target sample does not match ground truth"
                        f", and is also not complementary to it, hit_fraction={hit_fraction}"
                    )
                else:
                    print(f"{target_cram} - target sample can be normal-in-tumor sample, hit_fraction={hit_fraction}")
                    suspected_normal_in_tumor_crams.append(target_cram)
            else:
                print(f"{target_cram} - target sample match ground truth hit_fraction={hit_fraction}")

        # Validate that each normal cram ani-correlates the ground truth
        normal_germline_vcf_beds = []
        for normal_cram in self.normal_crams:
            hit_fraction = self.calc_hit_fraction(normal_cram, ground_truth_to_check)
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
            self.call_variants(normal_cram, normal_germline_vcf, hcr_in_region, min_count=self.min_count_germline_snps)
            self.vcf_to_bed(normal_germline_vcf, normal_germline_bed)
            normal_germline_vcf_beds.append(normal_germline_bed)

        # Validate that each normal-in-tumor sample matches at least one normal sample
        if len(self.normal_crams) > 0:
            for suspected_normal_in_tumor_cram in suspected_normal_in_tumor_crams:
                max_hit_fraction = 0
                best_match = ""
                for normal_germline_bed in normal_germline_vcf_beds:
                    hit_fraction = self.calc_hit_fraction(suspected_normal_in_tumor_cram, normal_germline_bed)
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

    def count_lines(self, in_file: str, out_file: str):
        self.sp.print_and_run(f"wc -l {in_file} " + "| awk '{print $1}' " + f" > {out_file}")

    def vcf_to_bed(self, vcf: str, bed: str, max_vars=10**8, region: str = "") -> None:
        sed_pattern = r"s/,<\*>//"
        self.sp.print_and_run(
            f"bcftools view -H --type snps {vcf} {region} | head -{max_vars}"
            " | awk -v OFS='\t' '{print $1,$2-1,$2,$5}'"
            f" | sed '{sed_pattern}' > {bed}"
        )

    def call_variants(self, cram: str, vcf: str, regions: str, min_count: int) -> None:
        local_cram = f"{vcf}.tmp.cram"
        self.sp.print_and_run(
            f"gatk PrintReads -I {cram} -L {regions} -R {self.ref} -O {local_cram} --verbosity WARNING"
        )
        self.sp.print_and_run(
            f"bcftools mpileup {local_cram} -f {self.ref} -T {regions} -a AD --skip-indels -d 500 "
            + f"| bcftools view -i 'AD[0:1]>={min_count}' -Oz -o {vcf}"
        )
        self.sp.print_and_run(f"bcftools index -t {vcf}")

    def calc_hit_fraction(self, cram: str, ground_truth_bed: str) -> float:
        cram_base_name = os.path.basename(cram)
        gt_base_name = f"{os.path.splitext(os.path.basename(ground_truth_bed))[0]}"
        vcf = f"{self.out_dir}/{cram_base_name}.hit.{gt_base_name}.vcf.gz"
        bed = f"{self.out_dir}/{cram_base_name}.hit.{gt_base_name}.bed"
        counts = f"{self.out_dir}/{cram_base_name}.hit.{gt_base_name}.counts"
        ground_truth_count_file = f"{self.out_dir}/{gt_base_name}.count"
        self.count_lines(ground_truth_bed, ground_truth_count_file)
        ground_truth_count = read_count(ground_truth_count_file)
        self.call_variants(cram, vcf, ground_truth_bed, min_count=self.min_count_snps)
        self.vcf_to_bed(vcf, bed)
        self.count_lines(bed, counts)
        hit_count = read_count(counts)
        hit_fraction = hit_count / ground_truth_count
        with open(f"{self.out_dir}/{cram_base_name}_{gt_base_name}.hit.txt", "w", encoding="utf-8") as fh:
            fh.write(f"hit_count {hit_count}\n")
            fh.write(f"hit_fraction {hit_fraction}\n")

        return hit_fraction


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
    bam_files = conf[f"{workflow_id}.bam_files"]
    background_bam_files = conf[f"{workflow_id}.background_bam_files"]
    ground_truth_vcf_files = conf[f"{workflow_id}.ground_truth_vcf_files"]
    training_hcr_files = conf[f"{workflow_id}.training_hcr_files"]
    training_intervals_files = conf[f"{workflow_id}.training_intervals"]

    region = args.region_str
    min_count_snps = args.min_count_snps
    min_count_germline_snps = args.min_count_germline_snps
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
                min_count_snps,
                min_count_germline_snps,
                min_hit_fraction_target,
                f"{args.out_dir}/subset_{i}",
                sp,
            ).check()
        )

    if len(errors) > 0:
        raise RuntimeError("\n".join(errors))