import sys

import pandas as pd
from pandas import DataFrame


def main():
    input_h5_file = sys.argv[1]
    output_prefix = sys.argv[2]

    bl_after_column = 'common_vars_hapmap_2_1_whole_genome_allele_1_again_filtered_002850_UGAv3_2_0.85-bwa'
    bl_before_column = 'common_vars_hapmap_2_1_whole_genome_allele_1_again'
    df: DataFrame = pd.read_hdf(input_h5_file) # df = pd.read_hdf(~/proj/error_correction/concordance/002850-UGAv3-2_40x.hcr_wgs_annot_not_consistency_allele_orig.h5)

    df['pos-1'] = df['pos'] - 1
    df['description'] = df['variant_type'] + '_' + df['hmer_indel_length'].astype(str)

    # was excluded from blacklist by SEC
    df['sec_call'] = ((df[bl_before_column]) & ~(df[bl_after_column]))


    initial_fn = df[df['classify'] == 'fn']
    initial_fp = df[df['classify'] == 'fp']
    initial_tp = df[df['classify'] == 'tp']
    initial_fn_indel = initial_fn[~initial_fn['indel_classify'].isna()]
    initial_fp_indel = initial_fp[~initial_fp['indel_classify'].isna()]
    initial_tp_indel = initial_tp[~initial_tp['indel_classify'].isna()]
    sec_fn_indel = initial_fn_indel[initial_fn_indel[bl_after_column] & (~initial_fn_indel['call'].isna())]
    sec_fp_indel = initial_fp_indel[initial_fp_indel['sec_call']]
    sec_tp_indel = initial_tp_indel[initial_tp_indel['sec_call']]
    unread_fn_indel = initial_fn_indel[initial_fn_indel['call'].isna()]

    write_bed(sec_fn_indel, f'{output_prefix}_sec_fn.bed')
    write_bed(sec_fp_indel, f'{output_prefix}_sec_fp.bed')
    write_bed(sec_tp_indel, f'{output_prefix}_sec_tp.bed')
    write_bed(unread_fn_indel, f'{output_prefix}_unread_fn.bed')


def write_bed(df: DataFrame, bed_path: str):
    df[['chrom', 'pos-1', 'pos', 'description']].to_csv(bed_path, index=False, sep='\t', header=None)


if __name__ == '__main__':
    main()




