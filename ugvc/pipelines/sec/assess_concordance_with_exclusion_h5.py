import argparse
import os
from os.path import splitext, basename, dirname
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from simppl.simple_pipeline import SimplePipeline

from python.pipelines.variant_filtering_utils import apply_filter
from python.pipelines.vcf_pipeline_utils import annotate_concordance
from ugvc import logger
from ugvc.concordance.concordance_utils import read_hdf, calc_accuracy_metrics, validate_and_preprocess_concordance_df

"""
Given a concordance h5 input, an exclude-list, and SEC refined exclude-list
Apply each exclude-list on the variants and measure the differences between the results.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--concordance_h5_input', help='path to h5 file describing comparison of variant calling to ground-truth',
        required=True)
    parser.add_argument(
        '--genome_fasta', help='path to fasta file of reference genome', required=True)
    parser.add_argument(
        '--raw_exclude_list', help='bed file containing raw exclude-list (SEC input)', required=True)
    parser.add_argument(
        '--sec_exclude_list', help=' bed file with sec_call_types written by SEC', required=True)
    parser.add_argument(
        '--dataset_key', help='chromosome name, in case h5 contains multiple datasets per chromosome', default='all')
    parser.add_argument('--hcr', help='bed file describing high confidence regions (runs.conservative.bed)',
                        required=True)
    parser.add_argument(
        '--output_prefix', help='prefix to output files containing stats and info about errors', required=True)
    parser.add_argument('--ignore_genotype', help='ignore genotype when comparing to ground-truth',
                        action='store_true', default=False)
    return parser.parse_args()


def write_status_bed_files(df: DataFrame,
                           output_prefix: str,
                           classification: Series) -> Tuple[str, str, str]:
    df['pos-1'] = df['pos'] - 1
    if 'sec_call_type' in df.columns:
        df['description'] = df['variant_type'] + '_' + df['hmer_indel_length'].astype(str) + '_' + df['sec_call_type']
    else:
        df['description'] = df['variant_type'] + '_' + df['hmer_indel_length'].astype(str)
    df.loc[df['description'].isna(), 'description'] = 'missing'
    fn = df[classification == 'fn']
    fp = df[classification == 'fp']
    tp = df[classification == 'tp']
    fn_indel = fn[~fn['indel_classify'].isna()]
    fp_indel = fp[~fp['indel_classify'].isna()]
    tp_indel = tp[~tp['indel_classify'].isna()]

    fn_file = f'{output_prefix}_fn_indel.bed'
    fp_file = f'{output_prefix}_fp_indel.bed'
    tp_file = f'{output_prefix}_tp_indel.bed'

    write_bed(fn_indel, fn_file)
    write_bed(fp_indel, fp_file)
    write_bed(tp_indel, tp_file)
    return fn_file, fp_file, tp_file


def write_bed(df: DataFrame, bed_path: str):
    df[['chrom', 'pos-1', 'pos', 'description']].to_csv(bed_path, index=False, sep='\t', header=None)


def main():
    args = parse_args()
    input_file = args.concordance_h5_input
    ref_genome_file = args.genome_fasta
    key = args.dataset_key

    classify_column = 'classify' if args.ignore_genotype else 'classify_gt'

    exclude_lists_beds = [args.raw_exclude_list, args.sec_exclude_list]
    out_pref = args.output_prefix

    os.makedirs(dirname(out_pref), exist_ok=True)

    logger.info(f'read_hdf: {key}')
    df: DataFrame = read_hdf(input_file, key=key)

    validate_and_preprocess_concordance_df(df)

    logger.info(f'annotate concordance with exclude_lists')
    df_annot, annots = annotate_concordance(df, ref_genome_file,
                                            runfile=args.hcr,
                                            flow_order='TGCA',
                                            annotate_intervals=exclude_lists_beds)
    stats_table = calc_accuracy_metrics(df_annot, classify_column)
    is_filtered = df_annot['filter'] != 'PASS'
    post_filter_classification = apply_filter(df_annot[classify_column], is_filtered)
    write_status_bed_files(df_annot, f'{out_pref}.original', post_filter_classification)

    with open(f'{out_pref}.original.stats.tsv', 'w') as stats_file:
        stats_file.write(f'{stats_table}\n')

    fp_files = [f'{out_pref}.original_fp_indel.bed']
    fn_files = [f'{out_pref}.original_fn_indel.bed']

    for i, exclude_list_bed_file in enumerate(exclude_lists_beds):
        exclude_list_name = splitext(basename(exclude_list_bed_file))[0]
        logger.info(f'exclude calls from {exclude_list_name}')

        if i == 0:
            exclude_list_df = pd.read_csv(exclude_list_bed_file, sep='\t', names=['chrom', 'pos-1', 'pos'])
            exclude_list_df.index = zip(exclude_list_df['chrom'], exclude_list_df['pos'])
        else:
            exclude_list_df = pd.read_csv(exclude_list_bed_file, sep='\t',
                                          names=['chrom', 'pos-1', 'pos', 'sec_call_type'])
            exclude_list_df.index = zip(exclude_list_df['chrom'], exclude_list_df['pos'])
            relevant_exclude_list_loci = set(exclude_list_df.index).intersection(set(df.index))
            # correct SEC filter, since non reference annotated positions should PASS SEC filter
            unfiltered = exclude_list_df[exclude_list_df['sec_call_type'] != 'reference']
            relevant_unfiltered_loci = set(df.index).intersection(set(unfiltered.index))
            df_annot.loc[relevant_unfiltered_loci, exclude_list_name] = False
            df_annot['sec_call_type'] = 'out_of_exclude_list'
            df_annot.loc[relevant_exclude_list_loci, 'sec_call_type'] = exclude_list_df.loc[relevant_exclude_list_loci, 'sec_call_type']

        # apply filter
        exclude_list_annot_df = df_annot.copy()
        exclude_list_annot_df.loc[df_annot[exclude_list_name], 'filter'] = ['BLACKLIST', 'SEC'][i]
        stats_table = calc_accuracy_metrics(exclude_list_annot_df, classify_column)
        is_filtered = exclude_list_annot_df['filter'] != 'PASS'
        post_filter_classification = apply_filter(exclude_list_annot_df[classify_column], is_filtered)

        with open(f'{out_pref}.{exclude_list_name}.stats.tsv', 'w') as stats_file:
            stats_file.write(f'{stats_table}\n')

        exclude_list_annot_df.to_hdf(f'{out_pref}.{exclude_list_name}.h5', key)
        fn_file, fp_file, tp_file = write_status_bed_files(exclude_list_annot_df, f'{out_pref}.{exclude_list_name}',
                                                           post_filter_classification)
        fp_files.append(fp_file)
        fn_files.append(fn_file)

    sp = SimplePipeline(0, 100, False)
    original_fp_file = fp_files[0]
    blacklist_fp_file = fp_files[1]
    sec_fp_file = fp_files[2]
    original_fn_file = fn_files[0]
    sec_fn_file = fn_files[2]
    for sct in ['non_noise_allele', 'uncorrelated', 'novel', 'known', 'unobserved']:
        sp.print_and_run(f'bedtools subtract -a {sec_fp_file} -b {blacklist_fp_file} '
                         f'| grep {sct} || true > {out_pref}.fp_missed.{sct}.bed')
    sp.print_and_run(f'bedtools subtract -a {original_fp_file} -b {sec_fp_file} > {out_pref}.fp_corrected.bed')
    sp.print_and_run(f'bedtools subtract -a {sec_fn_file} -b {original_fn_file} > {out_pref}.fn_added.bed')


if __name__ == '__main__':
    main()
