import argparse
import ast
from os.path import splitext, basename

import dask.dataframe as ddf
import numpy as np
import pandas as pd
from pandas import DataFrame

import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
from readwrite.read_from_gcs import read_hdf
from ugvc.concordance.concordance_utils import classify_variants, calculate_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--concordance_h5_input', help='path to h5 file describing comparison of variant calling to ground-truth',
        required=True)
    parser.add_argument(
        '--annotate_intervals', help='csv list of bed files to use as annotations', required=True)
    # 'exome.twist.bed',
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again.bed',
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa.bed',
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa.novel.bed'
    parser.add_argument(
        '--genome_fasta', help='path to fasta file of reference genome', required=True)
    parser.add_argument(
        '--exclude_lists', help='bed file to use as allele-specific exclude lists', required=True)
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again.bed',
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa.bed',
    # 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa.novel.bed'
    parser.add_argument(
        '--chr', help='chromosome name, in case h5 contains multiple datasets per chromosome', required=True)
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

def main():
    args = parse_args()
    input_file = args.concordance_h5_input  # '/data/mutect2/data_simulation/002850-UGAv3-2_40x.hcr_wgs.h5'  # output of comparison
    ref_genome_file = args.genome_fasta
    chrom = args.chr
    annotation_intervals = args.annotate_intervals.split(',')
    exclude_lists_beds = args.exclude_lists.split(',')
    out_pref = args.output_prefix

    df: DataFrame = read_hdf(input_file, key=chrom)

    df_annot, annots = vcf_pipeline_utils.annotate_concordance(df, ref_genome_file,
                                                               runfile='/data/runs.conservative.bed',
                                                               flow_order='TGCA',
                                                               annotate_intervals=annotation_intervals)

    df_annot = classify_variants(df_annot, ignore_gt=True)
    ddf_annot = ddf.from_pandas(df_annot, npartitions=30)
    called_alleles = ddf_annot.apply(extract_alleles, meta=('str'), axis=1).compute(scheduler='multiprocessing')
    df_annot['alleles_base'] = called_alleles

    for exclude_list_bed_file in exclude_lists_beds:
        exclude_list_name = splitext(basename(exclude_list_bed_file))[0]
        exclude_list_annot_df = df_annot.copy()
        exclude_list_df = pd.read_csv(exclude_list_bed_file, sep='\t', names=['chrom', 'pos', 'pos_1', 'alleles'])
        exclude_list_df['alleles'] = exclude_list_df['alleles'].apply(lambda x: np.array(ast.literal_eval(x)))
        exclude_list_df.index = zip(exclude_list_df['chrom'], exclude_list_df['pos_1'])

        is_in_bl = get_is_excluded_series(df_annot, exclude_list_df, exclude_list_name)

        exclude_list_annot_output_file = f'{out_pref}_{exclude_list_name}.h5'


        exclude_list_annot_df.loc[(exclude_list_annot_df['classify'] == 'tp') & (is_in_bl == True), 'classify'] = 'fn'
        exclude_list_annot_df = exclude_list_annot_df.loc[(exclude_list_annot_df['classify'] == 'fn') |
                                                                (is_in_bl == False)]
        exclude_list_annot_df.to_hdf(exclude_list_annot_output_file, 'wgs')

        exclude_list_annot_df = classify_variants(exclude_list_annot_df, ignore_gt=True)
        stats_table = calculate_results(exclude_list_annot_df)
        print(stats_table)

if __name__ == '__main__':
    main()