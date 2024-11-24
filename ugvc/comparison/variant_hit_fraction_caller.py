from __future__ import annotations

import os
import pandas as pd

from simppl.simple_pipeline import SimplePipeline
from ugvc.vcfbed.vcftools import get_vcf_df


def read_count(in_file: str) -> int:
    with open(in_file, encoding="utf-8") as f:
        return int(f.read().strip())


class VariantHitFractionCaller:
    def __init__(self, ref: str, out_dir: str, sp: SimplePipeline, min_af_snps: float, region: str):
        self.ref = ref
        self.out_dir = out_dir
        self.sp = sp
        self.min_af_snps = min_af_snps
        self.region = region

    def call_variants(self, cram: str, output_vcf: str, region: str, min_af: float) -> None:
        self.sp.print_and_run(
            f"bcftools mpileup {cram} -f {self.ref} -a ad,format/dp --skip-indels -d 500 "
            + f"| bcftools view -i 'AD[0:1] / format/DP >= {min_af}' -Oz -o {output_vcf}"
        )
        self.sp.print_and_run(f"bcftools index -t {output_vcf}")

    def calc_hit_fraction(self, called_vcf: str, ground_truth_vcf: str) -> tuple[float, float, float]:
        called_base_name = os.path.basename(called_vcf).replace(".vcf.gz", "")
        gt_base_name = os.path.basename(ground_truth_vcf).replace(".vcf.gz", "")
        ground_truth_variants = get_vcf_df(ground_truth_vcf)
        called_variants = get_vcf_df(called_vcf)
        # keep only major alt
        called_variants['major_alt'] = called_variants.apply(lambda x: x["alleles"][1], axis=1)
        ground_truth_variants['major_alt'] = ground_truth_variants.apply(lambda x: x["alleles"][1], axis=1)
        ground_truth_count = len(ground_truth_variants)
        hits = pd.merge(ground_truth_variants, called_variants, how="inner", on=["chrom", "pos", "ref", "major_alt"])
        hit_count = len(hits)
        hit_fraction = hit_count / (ground_truth_count + 0.001)
        if hit_fraction < 0.05:
            print(hits[['chrom', 'pos', 'ref', 'major_alt', 'alleles_x', 'alleles_y']])
        with open(f"{self.out_dir}/{called_base_name}_{gt_base_name}.hit.txt", "w", encoding="utf-8") as fh:
            fh.write(f"hit_count {hit_count}\n")
            fh.write(f"hit_fraction {hit_fraction}\n")

        return hit_fraction, hit_count, ground_truth_count

    def count_lines(self, in_file: str, out_file: str):
        self.sp.print_and_run(f"wc -l {in_file} " + "| awk '{print $1}' " + f" > {out_file}")

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
