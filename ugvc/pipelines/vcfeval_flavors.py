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
import json
import pysam

from simppl.simple_pipeline import SimplePipeline
from ugvc.utils.stats_utils import get_precision, get_recall, get_f1


def get_parser(argv: list[str]):
    ap = argparse.ArgumentParser(prog='vcfeval_flavors.py', description=run.__doc__)
    ap.add_argument('-b', '--baseline',
                        help="VCF file containing baseline variants", type=str, required=True)
    ap.add_argument('-c', '--calls',
                        help="VCF file containing called variants--output_prefix", type=str, required=True)
    ap.add_argument('-e', '--evaluation_regions',
                        help='if set, evaluate within regions contained in the supplied BED file, '
                             'allowing transborder matches. To be used for truth-set', type=str)
    ap.add_argument('-o', '--output',
                        help='directory for output', type=str, required=True)
    ap.add_argument('-t', '--template',
                        help='SDF of the reference genome the variants are called against')
    ap.add_argument('-p', '--allele_and_genotype_error_penalty',
                        type=int,
                        choices=[2, 1, 0, -1],
                        default=0,
                        help="-p 2: usual vcfeval, penalizes twice each wrong-allele/genotype errror\n" \
                             "-p 1: penalize wrong-allele/genotype only once (remove 0.5 of such fps and fns)\n" \
                             "-p 0: don't penalize wrong-allele/genotype (remove completely such fps and fns)\n" \
                             "-p -1: don't penalize, and also reward a tp for calling wrong-allele/genotype\n")
    return ap
    return args

def count_vcf_lines(vcf_file):
    v_iter = pysam.VariantFile(vcf_file)
    count = 0
    for variant in v_iter:
        count += 1
    return count

def run(argv: list[str]):
    """Evaluate VCF against baseline, giving alternative penalty to wrong-alleles and genotype errors"""
    parser = get_parser(argv)
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    sp = SimplePipeline(args.fc, args.lc, args.d)
    calls = args.calls
    baseline = args.baseline
    eval_reg = args.evaluation_regions
    sdf = args.template
    out_dir = args.output
    penalty = args.allele_and_genotype_error_penalty
    result = []
    sp.print_and_run(f'rm -rf {out_dir}')
    sp.print_and_run(f'rtg vcfeval -b {baseline} -c {calls} -e {eval_reg} -t {sdf} -o {out_dir}')
    sp.print_and_run(f'bedtools intersect -a {calls} -b {eval_reg} -header > {out_dir}/input_in_hcr.vcf')
    sp.print_and_run(f'bcftools view -f PASS {out_dir}/input_in_hcr.vcf > {out_dir}/input_in_hcr.pass.vcf')
    sp.print_and_run(f'bedtools intersect -a {baseline} -b {eval_reg} -header > {out_dir}/gtr_in_hcr.vcf')
    result.append('type tp fp fn precision recall f1')
    for vt in ['indels', 'snps']:
        for pref in ['input_in_hcr', 'gtr_in_hcr', 'input_in_hcr.pass']:
            sp.print_and_run(f'bcftools view {out_dir}/{pref}.vcf --type {vt} > {out_dir}/{pref}.{vt}.vcf')
        sp.print_and_run(f'bcftools view {baseline} --type {vt} > {out_dir}/gt.{vt}.vcf')
        for pref in ['fp', 'tp', 'fn']:
            sp.print_and_run(f'bcftools view {out_dir}/{pref}.vcf.gz --type {vt} > {out_dir}/{pref}.{vt}.vcf')
        sp.print_and_run(f'bedtools subtract -A -a {out_dir}/fp.{vt}.vcf -b {out_dir}/gt.{vt}.vcf -header >  {out_dir}/fp.{vt}.clean.vcf;')
        sp.print_and_run(f'bedtools subtract -A -a {out_dir}/fn.{vt}.vcf -b {out_dir}/input_in_hcr.pass.{vt}.vcf -header > {out_dir}/fn.{vt}.clean.vcf')

        tp = count_vcf_lines(f'{out_dir}/tp.{vt}.vcf')
        fp_clean = count_vcf_lines(f'{out_dir}/fp.{vt}.clean.vcf')
        fn_clean = count_vcf_lines(f'{out_dir}/fn.{vt}.clean.vcf');
        fp = count_vcf_lines(f'{out_dir}/fp.{vt}.vcf');
        fn = count_vcf_lines(f'{out_dir}/fn.{vt}.vcf');
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
        f1 = get_f1(precision / 100, recall / 100)  * 100
        result.append(f'{vt} {tp} {fp} {fn} {precision:.2f} {recall:.2f} {f1:.2f}')
    for line in result:
        print(line)

