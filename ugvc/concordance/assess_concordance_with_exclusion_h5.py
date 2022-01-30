import argparse
import ast
from os.path import splitext, basename

import dask.dataframe as ddf
import numpy as np
import pandas as pd
from pandas import DataFrame

from python.pipelines.vcf_pipeline_utils import annotate_concordance
from ugvc import logger
from ugvc.concordance.concordance_utils import classify_variants, calculate_results, read_hdf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--concordance_h5_input', help='path to h5 file describing comparison of variant calling to ground-truth',
        required=True)
    parser.add_argument(
        '--genome_fasta', help='path to fasta file of reference genome', required=True)
    parser.add_argument(
        '--initial_exclude_list', help='bed file of initial allele-specific exclude-list', required=True)
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again.bed',
    parser.add_argument(
        '--refined_exclude_lists', help='csv list of bed corrected exclude-lists by SEC', required=True)
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa.bed',
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa.novel.bed'
    parser.add_argument(
        '--chr', help='chromosome name, in case h5 contains multiple datasets per chromosome', required=True)
    parser.add_argument('--hcr', help='bed file describing high confidance regions (runs.conservative.bed)', required=True)
    parser.add_argument(
        '--output_prefix', help='prefix to output files containing stats and info about errors', required=True)
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


def write_exclusions_delta_bed_files(df: DataFrame,
                                     output_prefix: str,
                                     initial_exclude_list_name: str,
                                     refined_exclude_list_name: str):
    df['pos-1'] = df['pos'] - 1
    df['description'] = df['variant_type'] + '_' + df['hmer_indel_length'].astype(str)

    # was removed from initial exclude_list by refinement
    df['exclude_list_delta'] = ((df[initial_exclude_list_name]) & ~(df[refined_exclude_list_name]))

    initial_fn = df[df['classify'] == 'fn']
    initial_fp = df[df['classify'] == 'fp']
    initial_tp = df[df['classify'] == 'tp']
    initial_fn_indel = initial_fn[~initial_fn['indel_classify'].isna()]
    initial_fp_indel = initial_fp[~initial_fp['indel_classify'].isna()]
    initial_tp_indel = initial_tp[~initial_tp['indel_classify'].isna()]
    read_fn_indel = initial_fn_indel[initial_fn_indel[refined_exclude_list_name] & (initial_fn_indel['call'].isna())]
    delta_fp_indel = initial_fp_indel[initial_fp_indel['exclude_list_delta']]
    delta_tp_indel = initial_tp_indel[initial_tp_indel['exclude_list_delta']]
    unread_fn_indel = initial_fn_indel[initial_fn_indel['call'].isna()]

    write_bed(read_fn_indel, f'{output_prefix}_read_fn.bed')
    write_bed(delta_fp_indel, f'{output_prefix}_delta_fp.bed')
    write_bed(delta_tp_indel, f'{output_prefix}_delta_tp.bed')
    write_bed(unread_fn_indel, f'{output_prefix}_unread_fn.bed')


def write_bed(df: DataFrame, bed_path: str):
    df[['chrom', 'pos-1', 'pos', 'description']].to_csv(bed_path, index=False, sep='\t', header=None)


def main():
    args = parse_args()
    input_file = args.concordance_h5_input  # '/data/mutect2/data_simulation/002850-UGAv3-2_40x.hcr_wgs.h5'
    ref_genome_file = args.genome_fasta
    if args.chr:
        key = args.chr
    else:
        key = 'all'
    exclude_lists_beds = [args.initial_exclude_list] + args.refined_exclude_lists.split(',')
    out_pref = args.output_prefix

    logger.info(f'read_hdf: {key}')
    df: DataFrame = read_hdf(input_file, key=key)

    logger.info(f'annotate concordance with exclude_lists')
    df_annot, annots = annotate_concordance(df, ref_genome_file,
                                            runfile=args.hcr,
                                            flow_order='TGCA',
                                            annotate_intervals=exclude_lists_beds)

    logger.info('classify variants')
    df_annot = classify_variants(df_annot, ignore_gt=True)
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

        exclude_list_annot_df = df_annot.copy()
        exclude_list_annot_df.loc[is_in_bl == True, 'classify'] = 'fn'
        # mark excluded TP as FN
        exclude_list_annot_df.loc[(exclude_list_annot_df['classify'] == 'tp') & (is_in_bl == True), 'classify'] = 'fn'
        # remove excluded FP variants from table (true-negatives)
        exclude_list_annot_df = exclude_list_annot_df.loc[
            (exclude_list_annot_df['classify'] == 'fn') | (is_in_bl == False)]
        exclude_list_annot_df = classify_variants(exclude_list_annot_df, ignore_gt=True)
        stats_table = calculate_results(exclude_list_annot_df)

        with open(f'{out_pref}.{exclude_list_name}.stats.tsv', 'w') as stats_file:
            stats_file.write(f'{stats_table}\n')

        exclude_list_annot_df.to_hdf(f'{out_pref}.{exclude_list_name}.h5', key)
        if i > 0:
            write_exclusions_delta_bed_files(df=exclude_list_annot_df,
                                             output_prefix=f'{out_pref}.{exclude_list_name}',
                                             initial_exclude_list_name=initial_exclusion_list_name,
                                             refined_exclude_list_name=exclude_list_name)
        else:
            initial_exclusion_list_name = exclude_list_name


if __name__ == '__main__':
    main()
