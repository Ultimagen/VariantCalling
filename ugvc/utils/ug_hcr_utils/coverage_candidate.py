import os
import subprocess
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')

def low_map_regions(df_mean_std_all, min_cov_q0, min_frac):
    """
    calculate low mappability regions.
    Parameters
    ----------
    df_mean_std_all - dataframe with the following columns['chrom','chromStart','chromEnd','mean_q0','std_q0','mean_q20','std_q20']
    min_cov_q0 - minimal coverage for mapq0 reads
    min_frac - minimal fraction of mapq20 out of mapq0 reads.

    Returns
    -------
    dataframe with regions of low mappability.
    """
    df = df_mean_std_all[(df_mean_std_all['mean_q0'] > min_cov_q0) & (
                df_mean_std_all['mean_q20'] <= min_frac * df_mean_std_all['mean_q0'])]
    return df


def low_cov_regions(df_mean_std_all, min_cov_q20):
    """
    calculate low coverage regions.
    Parameters
    ----------
    df_mean_std_all - dataframe with the following columns['chrom','chromStart','chromEnd','mean_q0','std_q0','mean_q20','std_q20']
    min_cov_q20 - minimal coverage for mapq20 reads

    Returns
    -------
    dataframe with regions of low coverage.
    """
    df = df_mean_std_all[df_mean_std_all['mean_q20'] < min_cov_q20]
    return df


def high_cv_regions(df_mean_std_all, max_cv):
    """
    calculate high cv (std/mean) regions.
    Parameters
    ----------
    df_mean_std_all - dataframe with the following columns['chrom','chromStart','chromEnd','mean_q0','std_q0','mean_q20','std_q20']
    max_cv - maximal CV.

    Returns
    -------
    dataframe with regions of high CV.
    """
    df = df_mean_std_all
    df['std_div_mean_q20'] = df_mean_std_all['std_q20'] / df_mean_std_all['mean_q20']
    df['std_div_mean_q20'] = df['std_div_mean_q20'].fillna(0)
    df = df[df['std_div_mean_q20'] > max_cv]
    return df


def filer_gaps(bed_file, GAPS):
    """
    filter gaps from a given bed file
    Parameters
    ----------
    bed_file - input bed file
    GAPS - gaps bed file

    Returns
    -------
    filtered file <bed_file>.noGAPS.bed
    """
    out_file = bed_file + 'noGAPS.bed'
    cmd = f"bedtools subtract - a {bed_file} -b {GAPS} > {out_file}"
    subprocess.check_call(cmd, shell=True)
    return out_file


def generate_lcr_candidate(df_mean_std_all, q_comparison, min_cov_q0, min_frac, min_cov, max_cv, out_dir,GAPS):
    """
    generate coverage lcr candidate
    Parameters
    ----------
    df_mean_std_all - dataframe with the following columns['chrom','chromStart','chromEnd','mean_q0','std_q0','mean_q20','std_q20']
    q_comparison - mapq to compare to mapq0. we use mapq20.
    min_cov_q0 - minimal coverage for mapq0 reads for low mappability.
    min_frac - minimal fraction of mapq20 out of mapq0 for low mappability.
    min_cov - minimal coverage of mapq20 reads.
    max_cv - maximal CV (std/mean) for high CV regions
    out_dir - output directory

    Returns
    -------
    3 dataframes: low_map, low_cov, high_cv
    """

    low_map_df = low_map_regions(df_mean_std_all, min_cov_q0, min_frac)
    low_map_df[['chrom', 'chromStart', 'chromEnd']].to_csv(pjoin(out_dir, 'low_map.bed'), index=None, header=None,
                                                           sep='\t')
    low_map = filer_gaps(pjoin(out_dir, 'low_map.bed'), GAPS)

    # calculate_low_coverage_regions
    low_cov_df = low_cov_regions(df_mean_std_all, min_cov_q0)
    low_cov_df[['chrom', 'chromStart', 'chromEnd']].to_csv(pjoin(out_dir, 'low_cov.bed'), index=None, header=None,
                                                           sep='\t')
    low_cov = filer_gaps(pjoin(out_dir, 'low_cov.bed'), GAPS)

    # calculate_high_cv_regions
    high_cv_df = high_cv_regions(df_mean_std_all, max_cv)
    high_cv_df[['chrom', 'chromStart', 'chromEnd']].to_csv(pjoin(out_dir, 'high_cv.bed'), index=None, header=None,
                                                           sep='\t')
    high_cv = filer_gaps(pjoin(out_dir, 'high_cv.bed'), GAPS)

    return [low_map, low_cov, high_cv]


def write_candidate(low_map, low_cov, high_cv, dir_path):
    """
    writes lcr coverage candidate into a merged bed file
    Parameters
    ----------
    low_map - low mappability dataframe
    low_cov - low coverage dataframe
    high_cv - high CV dataframe
    dir_path - output dircetory

    Returns
    -------
    merged bed file :  dir_path/ug_cov.bed
    """
    cmd= f"bedtools sort -i {low_map} > {low_map}.sort.bed"
    subprocess.check_call(cmd, shell=True)

    cmd = f"bedtools sort -i {low_cov} > {low_cov}.sort.bed"
    subprocess.check_call(cmd, shell=True)

    cmd = f"bedtools sort - i {high_cv} > {high_cv}.sort.bed"
    subprocess.check_call(cmd, shell=True)

    intersection_file = pjoin(dir_path, 'lcr_intersections.bed')
    cmd = f"bedtools multiinter -names low_map low_cov high_cv \
        -i {low_map}.sort.bed \
        {low_cov}.sort.bed \
        {high_cv}.sort.bed \
        > {intersection_file}"
    subprocess.check_call(cmd, shell=True)

    with open(intersection_file) as infile:
        with open(pjoin(dir_path, "ug_cov.bed"), 'w') as outfile:
            for line in map(lambda x: x.split(), infile):
                name = line[4].replace(",", "|")
                outfile.write("\t".join((line[0], line[1], line[2], name)))
                outfile.write("\n")
    return os.path.join(dir_path, "ug_cov.bed")
