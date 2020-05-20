import pandas as pd
import numpy as np
from joblib import delayed, Parallel
import pysam
import gzip
import os


def _alt_or_ref_colname(c):
    if c == 0:
        return 'ref'
    else:
        return f'alt{c}'


def _expand_col(series_in, colname, by=',', mode='sample'):
    try:
        df_out = series_in.str.split(by, expand=True)
        if len(df_out.columns) == 1:
            df_out.columns = [colname]
        else:
            df_out.columns = [f'{colname}_{_alt_or_ref_colname(c) if mode == "sample" else c}' for c in df_out.columns]
    except AttributeError:
        return series_in
    return df_out


def _reformat_sample(series_in, sample_name, format_fields):
    df_tmp = pd.DataFrame(data=series_in.str.split(':', expand=True))
    df_tmp.columns = format_fields
    df_tmp = pd.concat(
        (_expand_col(df_tmp[c], c, by='/' if c == 'GT' else ',', mode='all' if c == 'GT' else 'sample') for c in
         df_tmp.columns), axis=1)
    df_tmp.columns = pd.MultiIndex.from_product(([sample_name], df_tmp.columns))
    df_tmp = df_tmp.replace(to_replace='.', value=np.nan).astype(float)
    return df_tmp


def _reformat_vcf(df_in, reformatted=True, multiprocessed_read=True):
    ind_format_col = list(df_in.columns).index('FORMAT')
    if reformatted:
        df_tmp = df_in.iloc[:, :ind_format_col].copy().astype({'POS': np.int, 'QUAL': np.float})
        df_variants = pd.concat((_expand_col(df_tmp[c], c, mode='all') for c in df_tmp.columns), axis=1)
        df_variants.columns = pd.MultiIndex.from_product((['variant'], df_variants.columns))
        format_fields = df_in['FORMAT'].iloc[0].split(':')
        df_out = pd.concat([df_variants] + list(Parallel(n_jobs=-1 if multiprocessed_read else 1)(
            delayed(_reformat_sample)(df_in[c], c, format_fields=format_fields) for c in
            df_in.columns[ind_format_col+1:])), axis=1
                           )
        df_out = df_out.replace(to_replace='.', value=np.nan).fillna(value=pd.np.nan)
    else:
        df_out = df_in

    sample_list = df_in.columns[ind_format_col+1:]
    return df_out, sample_list


def read_vcf_as_dataframe(vcf_file, reformatted=True, multiprocessed_read=True, **read_csv_kwargs):
    with pysam.VariantFile(vcf_file) as f_vcf:
        hdr = f_vcf.header
        header_lines = hdr.__str__().count(os.linesep)
    with gzip.open(vcf_file) as f_vcf:
        df = pd.read_csv(f_vcf, sep='\t', skiprows=header_lines - 1, **read_csv_kwargs).rename(
            columns={'#CHROM': 'CHROM'})

    return _reformat_vcf(df, reformatted, multiprocessed_read=multiprocessed_read)
