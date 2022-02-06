import argparse
import ast
from os.path import splitext, basename

import dask.dataframe as ddf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from python.pipelines.variant_filtering_utils import apply_filter
from python.pipelines.vcf_pipeline_utils import annotate_concordance
from ugvc import logger
from ugvc.concordance.concordance_utils import read_hdf, calc_accuracy_metrics, validate_and_preprocess_concordance_df

"""
Given a concordance h5 input, a blacklist (with alleles), and a list of SEC refined blacklists
Apply each blacklist (with allele-consistency) on the variants and measure the differences between the results.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--concordance_h5_input', help='path to h5 file describing comparison of variant calling to ground-truth',
        required=True)
    parser.add_argument(
        '--genome_fasta', help='path to fasta file of reference genome', required=True)
    parser.add_argument(
        '--initial_exclude_list', help='bed file of initial allele-specific exclude-list', required=True)
    parser.add_argument(
        '--refined_exclude_lists', help='csv list of bed corrected exclude-lists by SEC', required=True)
    parser.add_argument(
        '--dataset_key', help='chromosome name, in case h5 contains multiple datasets per chromosome', default='all')
    parser.add_argument('--hcr', help='bed file describing high confidance regions (runs.conservative.bed)', required=True)
    parser.add_argument(
        '--output_prefix', help='prefix to output files containing stats and info about errors', required=True)
    parser.add_argument('--ignore_genotype', help='ignore genotype when comparing to ground-truth',
                    action='store_true', default=False)
    return parser.parse_args()


def extract_alleles(x):
    if x['gt_ultima'] == (None,):
        return [x['alleles'][0]]
    return np.asarray(pd.Series(x['alleles']).loc[pd.Series(x['gt_ultima']).apply(lambda x: 0 if np.isnan(x) else x)])


def get_is_excluded_series(df_annot: DataFrame, exclude_list_df: DataFrame, exclude_list_name: str):
    is_in_bl = []
    for ind, variant in df_annot.iterrows():
        if (variant[exclude_list_name] == False) | (ind not in exclude_list_df.index):
            is_in_bl.append(False)
        else:
            bed_var = exclude_list_df.loc[[ind]]
            result = np.any(
                np.asarray(pd.Series(variant['alleles_base']).apply(lambda x: x in bed_var.iloc[0]['alleles'])))
            is_in_bl.append(result)
    is_in_bl = pd.Series(is_in_bl)
    is_in_bl.index = df_annot.index
    return is_in_bl


def write_status_bed_files(df: DataFrame,
                           output_prefix: str,
                           classification: Series):
    df['pos-1'] = df['pos'] - 1
    df['description'] = df['variant_type'] + '_' + df['hmer_indel_length'].astype(str)
    df.loc[df[df['description'].isna()].index, 'description'] = 'missing'
    initial_fn = df[classification == 'fn']
    initial_fp = df[classification == 'fp']
    initial_tp = df[classification == 'tp']
    initial_fn_indel = initial_fn[~initial_fn['indel_classify'].isna()]
    initial_fp_indel = initial_fp[~initial_fp['indel_classify'].isna()]
    initial_tp_indel = initial_tp[~initial_tp['indel_classify'].isna()]

    write_bed(initial_fn_indel, f'{output_prefix}_fn_indel.bed')
    write_bed(initial_fp_indel, f'{output_prefix}_fp_indel.bed')
    write_bed(initial_tp_indel, f'{output_prefix}_tp_indel.bed')


def write_bed(df: DataFrame, bed_path: str):
    df[['chrom', 'pos-1', 'pos', 'description']].to_csv(bed_path, index=False, sep='\t', header=None)


def main():
    args = parse_args()
    input_file = args.concordance_h5_input
    ref_genome_file = args.genome_fasta
    key = args.dataset_key

    classify_column = 'classify' if args.ignore_genotype else 'classify_gt'

    exclude_lists_beds = [args.initial_exclude_list] + args.refined_exclude_lists.split(',')
    out_pref = args.output_prefix

    logger.info(f'read_hdf: {key}')
    df: DataFrame = read_hdf(input_file, key=key)

    validate_and_preprocess_concordance_df(df)

    logger.info(f'annotate concordance with exclude_lists')
    df_annot, annots = annotate_concordance(df, ref_genome_file,
                                            runfile=args.hcr,
                                            flow_order='TGCA',
                                            annotate_intervals=exclude_lists_beds)

    write_status_bed_files(df_annot, f'{out_pref}.original', df_annot[classify_column])
    stats_table = calc_accuracy_metrics(df_annot, classify_column)
    with open(f'{out_pref}.original.stats.tsv', 'w') as stats_file:
        stats_file.write(f'{stats_table}\n')

    ddf_annot = ddf.from_pandas(df_annot, npartitions=30)

    logger.info('extract alleles')
    called_alleles = ddf_annot.apply(extract_alleles, meta='str', axis=1).compute(scheduler='multiprocessing')
    df_annot['alleles_base'] = called_alleles

    initial_exclusion_list_name = ''
    for i, exclude_list_bed_file in enumerate(exclude_lists_beds):
        exclude_list_name = splitext(basename(exclude_list_bed_file))[0]
        logger.info(f'exclude calls from {exclude_list_name}')

        exclude_list_df = pd.read_csv(exclude_list_bed_file, sep='\t', names=['chrom', 'pos', 'pos_1', 'alleles'])
        exclude_list_df['alleles'] = exclude_list_df['alleles'].apply(lambda x: np.array(ast.literal_eval(x)))
        exclude_list_df.index = zip(exclude_list_df['chrom'], exclude_list_df['pos_1'])

        is_in_bl = get_is_excluded_series(df_annot, exclude_list_df, exclude_list_name)

        # remove non_matching_alleles positions from exclude-list
        exclude_list_annot_df = df_annot.copy()
        exclude_list_annot_df.loc[df_annot[is_in_bl].index, 'filter'] = 'BLACKLIST'
        stats_table = calc_accuracy_metrics(exclude_list_annot_df, classify_column)
        is_filtered = exclude_list_annot_df['filter'] != 'PASS'
        post_filter_classification = apply_filter(exclude_list_annot_df[classify_column], is_filtered)

        with open(f'{out_pref}.{exclude_list_name}.stats.tsv', 'w') as stats_file:
            stats_file.write(f'{stats_table}\n')

        exclude_list_annot_df.to_hdf(f'{out_pref}.{exclude_list_name}.h5', key)
        if i > 0:
            write_status_bed_files(exclude_list_annot_df, f'{out_pref}.{exclude_list_name}', post_filter_classification)


if __name__ == '__main__':
    main()
