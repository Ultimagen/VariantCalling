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
#    Compare vcf file to ground-truth in alternative manner
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import subprocess

from simppl.simple_pipeline import SimplePipeline
from ugbio_comparison import vcf_comparison_utils as vpc
from ugbio_core import vcf_utils as vpu
from ugbio_core.vcfbed.interval_file import IntervalFile

from ugvc.utils.stats_utils import get_f1, get_precision, get_recall


def get_parser():
    ap = argparse.ArgumentParser(prog="vcfeval_flavors.py", description=run.__doc__)
    ap.add_argument("-b", "--baseline", help="VCF file containing baseline variants", type=str, required=True)
    ap.add_argument("-c", "--calls", help="VCF file containing called variants--output_prefix", type=str, required=True)
    ap.add_argument(
        "-e",
        "--evaluation_regions",
        help="if set, evaluate within regions contained in the intersection of the supplied bed files,"
        " allowing transborder matches. To be used for truth-set",
        action="append",
        type=str,
        default=[],
    )
    ap.add_argument(
        "--evaluation_intervals",
        help="if set, intersect evaluation_regions with interval_list comma-separated files list",
        action="append",
        type=str,
        default=[],
    )
    ap.add_argument("-o", "--output", help="directory for output", type=str, required=True)
    ap.add_argument("-t", "--template", help="SDF of the reference genome the variants are called against")
    ap.add_argument(
        "-p",
        "--allele_and_genotype_error_penalty",
        type=int,
        choices=[2, 1, 0, -1],
        default=1,
        help="-p 2: usual vcfeval, penalizes twice each wrong-allele/genotype errror\n"
        "-p 1: penalize wrong-allele/genotype only once (remove 0.5 of such fps and fns)\n"
        "-p 0: don't penalize wrong-allele/genotype (remove completely such fps and fns)\n"
        "-p -1: don't penalize, and also reward a tp for calling wrong-allele/genotype\n",
    )
    ap.add_argument(
        "--var_type",
        type=str,
        choices=["snps", "indels", "both"],
        default="both",
        help="choose to analyze specific variant type (indels is much faster than snps)",
    )
    return ap


def count_vcf_lines(vcf_file):
    return int(subprocess.check_output(f"bcftools index -n {vcf_file}".split()))


def run(argv: list[str]):
    """Evaluate VCF against baseline, giving alternative penalty to wrong-alleles and genotype errors"""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    print(argv)
    args = parser.parse_args(argv[1:])
    sp = SimplePipeline(args.fc, args.lc, args.d)
    calls = args.calls
    baseline = args.baseline
    eval_regions_files: list[str] = args.evaluation_regions
    eval_intervals_files: list[str] = args.evaluation_intervals
    sdf = args.template
    out_dir = args.output
    penalty = args.allele_and_genotype_error_penalty
    variant_types = ["indels", "snps"] if args.var_type == "both" else [args.var_type]
    result = []
    sp.print_and_run(f"rm -rf {out_dir}")
    sp.print_and_run(f"mkdir -p {out_dir}")

    # convert interval_list files to bed files
    for eval_intervals_file in eval_intervals_files:
        eval_regions_files.append(IntervalFile(sp, eval_intervals_file).as_bed_file())

    first_eval_regions_file = eval_regions_files[0]
    intersected_eval_regions = first_eval_regions_file
    vpe = vpu.VcfUtils(sp)
    vpcp = vpc.VcfComparisonUtils(sp)

    for i, eval_regions_file in enumerate(eval_regions_files[1:]):
        intersected_eval_regions = f"{out_dir}/eval_regions.{i}.bed"
        vpe.intersect_bed_files(first_eval_regions_file, eval_regions_file, intersected_eval_regions)
        first_eval_regions_file = intersected_eval_regions
    eval_reg = intersected_eval_regions
    sp.print_and_run(f"bcftools view {calls} -R {eval_reg} -Oz > {out_dir}/input_in_hcr.vcf.gz")
    sp.print_and_run(f"bcftools view -f PASS,. {out_dir}/input_in_hcr.vcf.gz -Oz > {out_dir}/input_in_hcr.pass.vcf.gz")
    vpe.index_vcf(f"{out_dir}/input_in_hcr.pass.vcf.gz")
    sp.print_and_run(f"bcftools view {baseline} -R {eval_reg} -Oz > {out_dir}/gtr_in_hcr.vcf.gz")
    vpe.index_vcf(f"{out_dir}/gtr_in_hcr.vcf.gz")
    vpcp.run_vcfeval(
        f"{out_dir}/input_in_hcr.pass.vcf.gz", f"{out_dir}/gtr_in_hcr.vcf.gz", eval_reg, f"{out_dir}/vcfeval", sdf
    )
    result.append("type tp fp fn precision recall f1")
    for vt in variant_types:
        for pref in ("input_in_hcr", "gtr_in_hcr", "input_in_hcr.pass"):
            sp.print_and_run(f"bcftools view {out_dir}/{pref}.vcf.gz --type {vt} -Oz > {out_dir}/{pref}.{vt}.vcf.gz")
            vpe.index_vcf(f"{out_dir}/{pref}.{vt}.vcf.gz")

        sp.print_and_run(f"bcftools view {baseline} --type {vt} -Oz > {out_dir}/gt.{vt}.vcf.gz")
        vpe.index_vcf(f"{out_dir}/gt.{vt}.vcf.gz")

        for pref in ("fp", "tp", "fn"):
            sp.print_and_run(
                f"bcftools view {out_dir}/vcfeval/{pref}.vcf.gz --type {vt} -Oz > {out_dir}/{pref}.{vt}.vcf.gz"
            )
            vpe.index_vcf(f"{out_dir}/{pref}.{vt}.vcf.gz")
        sp.print_and_run(
            f"bcftools isec -C {out_dir}/fp.{vt}.vcf.gz {out_dir}/gt.{vt}.vcf.gz -Oz -w 1 "
            f"-o {out_dir}/fp.{vt}.clean.vcf.gz"
        )
        vpe.index_vcf(f"{out_dir}/fp.{vt}.clean.vcf.gz")
        sp.print_and_run(
            f"bcftools isec -C {out_dir}/fn.{vt}.vcf.gz "
            f"{out_dir}/input_in_hcr.pass.{vt}.vcf.gz -Oz -w 1 -o {out_dir}/fn.{vt}.clean.vcf.gz"
        )
        vpe.index_vcf(f"{out_dir}/fn.{vt}.clean.vcf.gz")
        tp = count_vcf_lines(f"{out_dir}/tp.{vt}.vcf.gz")
        fp_clean = count_vcf_lines(f"{out_dir}/fp.{vt}.clean.vcf.gz")
        fn_clean = count_vcf_lines(f"{out_dir}/fn.{vt}.clean.vcf.gz")
        fp = count_vcf_lines(f"{out_dir}/fp.{vt}.vcf.gz")
        fn = count_vcf_lines(f"{out_dir}/fn.{vt}.vcf.gz")
        allele_and_genotype_fp_errors = fp - fp_clean
        allele_and_genotype_fn_errors = fn - fn_clean

        if penalty == 1:
            fp -= allele_and_genotype_fp_errors / 2
            fn -= allele_and_genotype_fn_errors / 2
        elif penalty == 0:
            fp -= allele_and_genotype_fp_errors
            fn -= allele_and_genotype_fn_errors
        elif penalty == -1:
            fp -= allele_and_genotype_fp_errors
            fn -= allele_and_genotype_fn_errors
            tp += (allele_and_genotype_fp_errors + allele_and_genotype_fn_errors) / 2
        precision = get_precision(fp, tp) * 100
        recall = get_recall(fn, tp) * 100
        f1 = get_f1(precision / 100, recall / 100) * 100
        result.append(f"{vt} {tp} {fp} {fn} {precision:.2f} {recall:.2f} {f1:.2f}")
    for line in result:
        print(line)
    return result
