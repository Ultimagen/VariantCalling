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
from ugvc.utils.stats_utils import get_recall, get_precision

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
    indels = df['indel']
    fn_indel = df[classification == 'fn']
    fp_indel = df[classification == 'fp']
    tp_indel = df[classification == 'tp']

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
    stats_table.to_csv(f'{out_pref}.original.stats.csv', sep=';', index=False)
    # Apply original filters 
    df_annot[classify_column] = apply_filter(df_annot[classify_column], df_annot['filter'] != 'PASS')
    write_status_bed_files(df_annot, f'{out_pref}.original', df_annot[classify_column])


    for i, exclude_list_bed_file in enumerate(exclude_lists_beds):
        exclude_list_name = splitext(basename(exclude_list_bed_file))[0]
        logger.info(f'exclude calls from {exclude_list_name}')

        exclude_list_df = None
        if i == 1:
            exclude_list_df = pd.read_csv(exclude_list_bed_file, sep='\t',
                                          names=['chrom', 'pos-1', 'pos', 'sec_call_type', 'spv'])
            exclude_list_df.index = zip(exclude_list_df['chrom'], exclude_list_df['pos'])
            relevant_exclude_list_loci = exclude_list_df.index.intersection(df.index)
            # correct SEC filter, since non reference annotated positions should PASS SEC filter
            filtered = exclude_list_df[exclude_list_df['sec_call_type'] == 'reference']
            relevant_filtered_loci = df.index.intersection(filtered.index)
            df_annot[exclude_list_name] = False
            df_annot.loc[relevant_filtered_loci, exclude_list_name] = True
            df_annot['sec_call_type'] = 'out_of_exclude_list'
            df_annot.loc[relevant_exclude_list_loci, 'sec_call_type'] = \
                exclude_list_df.loc[relevant_exclude_list_loci, 'sec_call_type']

        # apply filter
        exclude_list_annot_df = df_annot.copy()
        exclude_list_annot_df.loc[df_annot[exclude_list_name], 'filter'] = ['BLACKLIST', 'SEC'][i]
        stats_table = calc_accuracy_metrics(exclude_list_annot_df, classify_column)
        is_filtered = exclude_list_annot_df['filter'] != 'PASS'
        exclude_list_annot_df[classify_column] = apply_filter(exclude_list_annot_df[classify_column], is_filtered)

        if i == 1:
            sec_call_types = {'novel', 'known', 'unobserved'}
            sec_pass_tp_df = exclude_list_annot_df[(df_annot[classify_column] == 'tp') & (exclude_list_annot_df['sec_call_type'].isin(sec_call_types))]
            sec_filter_tp_df = exclude_list_annot_df[(df_annot[classify_column] == 'tp') & (exclude_list_annot_df['sec_call_type'] == 'reference')]
            sec_pass_fp_df = exclude_list_annot_df[(df_annot[classify_column] == 'fp') & (exclude_list_annot_df['sec_call_type'].isin(sec_call_types))]
            sec_filter_fp_df = exclude_list_annot_df[(df_annot[classify_column] == 'fp') & (exclude_list_annot_df['sec_call_type'] == 'reference')]
            
            with open(f'{out_pref}.sec.stats.csv', 'w') as fh:
                fh.write('variant_type,filtered-tp,passed-tp,filtered-fp,passed-fp,SEC-recall,SEC-precision\n')
                sec_dfs = [sec_pass_tp_df, sec_filter_tp_df, sec_pass_fp_df, sec_filter_fp_df]
                for variant_type in ['snp', 'h-indel', 'non-h-indel']:
                    sec_dfs_var_type = [df[df['variant_type'] == variant_type] for df in sec_dfs]
                    sec_pass_tp, sec_filter_tp, sec_pass_fp, sec_filter_fp = (df.shape[0] for df in sec_dfs_var_type)
                    sec_recall = get_recall(fn=sec_filter_tp, tp=sec_pass_tp)
                    sec_precision = get_precision(fp=sec_pass_fp, tp=sec_filter_fp)
                    fh.write(f'{variant_type},{sec_filter_tp},{sec_pass_tp},{sec_filter_fp},{sec_pass_fp},{sec_recall},{sec_precision}\n')
                    write_bed(sec_dfs_var_type[0], f'{out_pref}.{variant_type}.sec_passed_tp.bed')
                    write_bed(sec_dfs_var_type[1], f'{out_pref}.{variant_type}.sec_filtered_tp.bed')
                    write_bed(sec_dfs_var_type[2], f'{out_pref}.{variant_type}.sec_passed_fp.bed')
                    write_bed(sec_dfs_var_type[3], f'{out_pref}.{variant_type}.sec_filtered_fp.bed')
            with open(f'{out_pref}.all_types.stats.csv', 'w') as fh:
                fh.write('sec_call_type,variant_type,passed-tp,passed-fp,precision\n')
                for sec_call_type in ['non_noise_allele', 'uncorrelated', 'unobserved', 'known', 'novel']:
                    for variant_type in ['snp', 'h-indel', 'non-h-indel']:
                        pass_tp_df = exclude_list_annot_df[(df_annot[classify_column] == 'tp') & (exclude_list_annot_df['sec_call_type'] == sec_call_type)]
                        pass_fp_df = exclude_list_annot_df[(df_annot[classify_column] == 'fp') & (exclude_list_annot_df['sec_call_type'] == sec_call_type)]
                        pass_tp_df, pass_fp_df = (df[df['variant_type'] == variant_type] for df in [pass_tp_df, pass_fp_df])
                        pass_tp = pass_tp_df.shape[0]
                        pass_fp = pass_fp_df.shape[0]
                        precision =  get_precision(fp=pass_fp, tp=pass_tp)
                        fh.write(f'{sec_call_type},{variant_type},{pass_tp},{pass_fp},{precision}\n')
                        write_bed(pass_tp_df, '{out_pref}.{variant_type}.{sec_call_type}_passed_tp.bed')
                        write_bed(pass_fp_df, '{out_pref}.{variant_type}.{sec_call_type}_passed_fp.bed')

        stats_table.to_csv(f'{out_pref}.{exclude_list_name}.stats.csv', sep=';', index=False)
        exclude_list_annot_df.to_hdf(f'{out_pref}.{exclude_list_name}.h5', key)



if __name__ == '__main__':
    main()
