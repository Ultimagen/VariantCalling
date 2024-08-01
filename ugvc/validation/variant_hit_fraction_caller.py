from __future__ import annotations

import os

from simppl.simple_pipeline import SimplePipeline


def read_count(in_file: str) -> int:
    with open(in_file, encoding="utf-8") as f:
        return int(f.read().strip())


class VariantHitFractionCaller:
    def __init__(self, ref: str, out_dir: str, sp: SimplePipeline, min_af_snps: float):
        self.ref = ref
        self.out_dir = out_dir
        self.sp = sp
        self.min_af_snps = min_af_snps

    def call_variants(self, cram: str, vcf: str, regions: str, min_af: float) -> None:
        self.sp.print_and_run(
            f"bcftools mpileup {cram} -f {self.ref} -T {regions} -a ad,format/dp --skip-indels -d 500 "
            + f"| bcftools view -i 'AD[0:1] / format/DP >= {min_af}' -Oz -o {vcf}"
        )
        self.sp.print_and_run(f"bcftools index -t {vcf}")

    def calc_hit_fraction(self, cram: str, ground_truth_bed: str) -> tuple[float, float, float]:
        cram_base_name = os.path.basename(cram)
        gt_base_name = f"{os.path.splitext(os.path.basename(ground_truth_bed))[0]}"
        vcf = f"{self.out_dir}/{cram_base_name}.hit.{gt_base_name}.vcf.gz"
        bed = f"{self.out_dir}/{cram_base_name}.hit.{gt_base_name}.bed"
        counts = f"{self.out_dir}/{cram_base_name}.hit.{gt_base_name}.counts"
        ground_truth_count_file = f"{self.out_dir}/{gt_base_name}.count"
        self.count_lines(ground_truth_bed, ground_truth_count_file)
        ground_truth_count = read_count(ground_truth_count_file)
        self.call_variants(cram, vcf, ground_truth_bed, min_af=self.min_af_snps)
        self.vcf_to_bed(vcf, bed)
        self.count_lines(bed, counts)
        hit_count = read_count(counts)
        hit_fraction = hit_count / (ground_truth_count + 0.001)
        with open(f"{self.out_dir}/{cram_base_name}_{gt_base_name}.hit.txt", "w", encoding="utf-8") as fh:
            fh.write(f"hit_count {hit_count}\n")
            fh.write(f"hit_fraction {hit_fraction}\n")

        return hit_fraction, hit_count, ground_truth_count

    def count_lines(self, in_file: str, out_file: str):
        self.sp.print_and_run(f"wc -l {in_file} " + "| awk '{print $1}' " + f" > {out_file}")

    def vcf_to_bed(self, vcf: str, bed: str, max_vars=10**8, region: str = "") -> None:
        sed_pattern = r"s/,<\*>//"
        self.sp.print_and_run(
            f"bcftools view -H --type snps {vcf} {region} | head -{max_vars}"
            " | awk -v OFS='\t' '{print $1,$2-1,$2,$5}'"
            f" | sed '{sed_pattern}' > {bed}"
        )

    @staticmethod
    def add_args_to_parser(parser):
        parser.add_argument(
            "--max_vars", type=int, default=2000, help="max number of variants to check for concordance"
        )
        parser.add_argument(
            "--min_af_snps", type=float, default=0.03, help="min allele frequency to count as a ground-truth hit"
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
