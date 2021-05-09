import pandas as pd
import numpy as np
from joblib import delayed, Parallel
import pysam
import gzip
import os
from os.path import join as pjoin, dirname
from multiprocessing import cpu_count
import tempfile
from .utils import run_shell_command, get_tagged_output_file
from pandas.api.types import CategoricalDtype

CHROM_DTYPE = CategoricalDtype(
    categories=[f"chr{j}" for j in range(1, 23)] + ["chrX", "chrY", "chrM"], ordered=True
)


def _alt_or_ref_colname(c):
    if c == 0:
        return "ref"
    else:
        return f"alt{c}"


def _expand_col(series_in, colname, by=",", mode="sample"):
    try:
        df_out = series_in.str.split(by, expand=True)
        if len(df_out.columns) == 1:
            df_out.columns = [colname]
        else:
            df_out.columns = [
                f'{colname}_{_alt_or_ref_colname(c) if mode == "sample" else c}'
                for c in df_out.columns
            ]
    except AttributeError:
        return series_in
    return df_out


def _reformat_sample(series_in, sample_name, format_fields):
    df_tmp = pd.DataFrame(data=series_in.str.split(":", expand=True))
    df_tmp.columns = format_fields
    df_tmp = pd.concat(
        (
            _expand_col(
                df_tmp[c],
                c,
                by="/" if c == "GT" else ",",
                mode="all" if c == "GT" else "sample",
            )
            for c in df_tmp.columns
        ),
        axis=1,
    )
    df_tmp.columns = pd.MultiIndex.from_product(([sample_name], df_tmp.columns))
    df_tmp = df_tmp.replace(to_replace=".", value=np.nan).astype(float)
    return df_tmp


def _reformat_vcf(df_in, reformatted=True, multiprocessed_read=True):
    df_in = df_in.astype(
        {
            "CHROM": "category",
            "POS": np.int32,
            "REF": "category",
            "FILTER": "category",
            "QUAL": np.float,
        }
    )
    df_in = df_in.astype({x: "category" for x in df_in.filter(regex="ALT").columns})
    df_in.columns = [x.lower() for x in df_in.columns]
    df_in = df_in.set_index(["chrom", "pos"])

    ind_format_col = list(df_in.columns).index("format")
    if reformatted:
        df_tmp = df_in.iloc[:, :ind_format_col].copy()
        df_variants = pd.concat(
            (_expand_col(df_tmp[c], c, mode="all") for c in df_tmp.columns), axis=1
        )
        df_variants.columns = pd.MultiIndex.from_product(
            (["variant"], df_variants.columns)
        )
        format_fields = df_in["format"].iloc[0].split(":")
        df_out = pd.concat(
            [df_variants]
            + list(
                Parallel(n_jobs=-1 if multiprocessed_read else 1)(
                    delayed(_reformat_sample)(df_in[c], c, format_fields=format_fields)
                    for c in df_in.columns[ind_format_col + 1 :]
                )
            ),
            axis=1,
        )
        df_out = df_out.replace(to_replace=".", value=np.nan).fillna(value=pd.np.nan)
    else:
        df_out = df_in

    sample_list = df_in.columns[ind_format_col + 1 :]
    return df_out, sample_list


def read_vcf_as_dataframe(
    vcf_file, reformatted=False, multiprocessed_read=True, **read_csv_kwargs
):
    with pysam.VariantFile(vcf_file) as f_vcf:
        hdr = f_vcf.header
        header_lines = hdr.__str__().count(os.linesep)
    with gzip.open(vcf_file) as f_vcf:
        df = pd.read_csv(
            f_vcf, sep="\t", skiprows=header_lines - 1, **read_csv_kwargs
        ).rename(columns={"#CHROM": "CHROM"})

    return _reformat_vcf(df, reformatted, multiprocessed_read=multiprocessed_read)


def index_vcfs(in_path, print_output=True, ext=".vcf.gz", n_threads=None):
    if n_threads is None:
        n_threads = cpu_count()
    if in_path.endswith(ext):
        if not os.path.isfile(in_path):
            raise ValueError(f"{in_path} does not exist - no such file")
        if print_output:
            print(f"indexing {in_path}")
        if not os.path.isfile(in_path + ".csi"):
            out = run_shell_command(f"bcftools index --threads {n_threads} {in_path}")
            if print_output and out is not None and len(out) > 0:
                print(out)
    else:
        if not os.path.isdir(in_path):
            raise ValueError(f"{in_path} does not exist - no such directory")
        for dirpath, dirnames, filenames in os.walk(in_path):
            for f in filenames:
                if f.endswith(ext):
                    index_vcfs(pjoin(dirpath, f), print_output=print_output)


def compress_vcfs(in_path, print_output=True, ext=".vcf", n_threads=None):
    if n_threads is None:
        n_threads = cpu_count()
    if in_path.endswith(ext):
        if not os.path.isfile(in_path):
            raise ValueError(f"{in_path} does not exist - no such file")
        if not os.path.isfile(in_path + ".gz"):
            out = run_shell_command(
                f"bcftools view {in_path} --threads {n_threads} -Oz -o {in_path + '.gz'}"
            )
            if print_output and out is not None and len(out) > 0:
                print(out)
    else:
        if not os.path.isdir(in_path):
            raise ValueError(f"{in_path} does not exist - no such directory")
        for dirpath, dirnames, filenames in os.walk(in_path):
            for f in filenames:
                if f.endswith(ext):
                    compress_vcfs(pjoin(dirpath, f), print_output=print_output)


def filter_vcfs(
    in_path,
    ext_expected=".vcf.gz",
    tag="filtered",
    min_ad_alt1_in_vcf=None,
    cmd_args=None,
    print_output=True,
    recursive=False,
):
    if cmd_args is None:
        cmd_args = f"--types snps --novel"
        if min_ad_alt1_in_vcf is not None:
            cmd_args += f" -i'FORMAT/AD[:1]>{min_ad_alt1_in_vcf - 1}'"
    if in_path.endswith(ext_expected) and not in_path.endswith(f".{tag}{ext_expected}"):
        if not os.path.isfile(in_path):
            raise ValueError(f"{in_path} does not exist - no such file")
        output_file = get_tagged_output_file(tag, in_path,)
        out = run_shell_command(
            f"bcftools view {cmd_args} {in_path} -O z -o {output_file}"
        )
        if print_output and out is not None and len(out) > 0:
            print(out)
        return output_file
    else:
        for dirpath, dirnames, filenames in os.walk(in_path):
            if not recursive and (dirpath != in_path):
                continue
            for f in filenames:
                if f.endswith(ext_expected):
                    filter_vcfs(pjoin(dirpath, f), print_output=print_output)


def concat_vcfs(in_files, output_file, also_index=True, print_output=False):
    run_shell_command(
        f"bcftools concat {' '.join([f + '.gz' for f in in_files])} -O z -o {output_file}"
    )
    if also_index:
        index_vcfs(output_file, print_output=print_output)


def intersect_vcfs(input_file, output_file, complement_file, collapse_mode="snps"):
    with tempfile.TemporaryDirectory(dir=dirname(output_file)) as tmpdirname:
        run_shell_command(
            f"bcftools isec --collapse {collapse_mode} --complement {input_file} {complement_file} -O z -p {tmpdirname}"
        )
        isec_output = pjoin(tmpdirname, "0000.vcf.gz")
        if os.path.isfile(isec_output):
            os.rename(isec_output, output_file)
        if os.path.isfile(isec_output + ".tbi"):
            os.rename(isec_output + ".tbi", output_file + ".tbi")
