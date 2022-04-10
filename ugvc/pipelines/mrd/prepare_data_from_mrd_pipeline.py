#!/env/python
import argparse
import os
import sys
from os.path import join as pjoin
from typing import List

from ugvc.mrd.mrd_utils import read_intersection_dataframes, read_signature
from ugvc.utils.consts import FileExtension


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prepare_data", description=run.__doc__)
    parser.add_argument(
        "-f",
        "--intersected-featuremaps",
        nargs="+",
        type=str,
        required=True,
        help="Input signature and featuemaps vcf files",
    )
    parser.add_argument(
        "-s",
        "--signature-vcf",
        nargs="+",
        type=str,
        required=True,
        help="Input signature vcf files",
    )
    parser.add_argument(
        "-e",
        "--snp-error-rate",
        type=str,
        default=None,
        required=False,
        help="SNP error rate hdf produced by mrd.snp_error_rate",
    )
    parser.add_argument(
        "-c",
        "--coverage-bw",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Coverage bigwig files generated with 'coverage_analysis full_analysis'",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        required=False,
        default="tumor",
        help=""" sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to
    be the full sample name, but does have to return a unique result. Default: "tumor" """,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="""Path to which output files will be written.""",
    )
    parser.add_argument(
        "-b",
        "--output-basename",
        type=str,
        default=None,
        help="""Basename of output files that will be created.""",
    )
    return parser.parse_args(argv)


def run(argv: List[str]):
    """Aggregate the outputs from the MRDFeatureMap pipeline and prepare dataframes for analysis"""
    args_in = parse_args(argv)
    prepare_data_from_mrd_pipeline(
        intersected_featuremaps_parquet=args_in.intersected_featuremaps,
        signature_vcf_files=args_in.signature_vcf,
        snp_error_rate_hdf=args_in.snp_error_rate,
        coverage_bw_files=args_in.coverage_bw,
        sample_name=args_in.sample_name,
        output_dir=args_in.output_dir,
        output_basename=args_in.output_basename,
    )
    sys.stdout.write("DONE" + os.linesep)


def prepare_data_from_mrd_pipeline(
    intersected_featuremaps_parquet,
    signature_vcf_files,
    snp_error_rate_hdf=None,
    coverage_bw_files=None,
    sample_name="tumor",
    output_dir=None,
    output_basename=None,
):
    """

    intersected_featuremaps_parquet
        list of featuremaps intesected with various signatures
    signature_vcf_files
        File name or a list of file names
    snp_error_rate_hdf
        snp error rate result generated from mrd.snp_error_rate, disabled (None) by default
    coverage_bw_files
        Coverage bigwig files generated with "coverage_analysis full_analysis", disabled (None) by default
    sample_name
        sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to be the
        full sample name, but does have to return a unique result. Default: "tumor"
    output_dir
        path to which output will be written if not None (default None)
    output_basename
        basename of output file (if output_dir is not None must also be not None), default None

    Returns
    -------
    dataframe
    """
    if output_dir is not None and output_basename is None:
        raise ValueError(
            f"output_dir is not None ({output_dir}) but output_basename is"
        )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    intersection_dataframe_fname = (
        pjoin(output_dir, f"{output_basename}.features{FileExtension.PARQUET.value}")
        if output_dir is not None
        else None
    )
    signatures_dataframe_fname = (
        pjoin(output_dir, f"{output_basename}.signatures{FileExtension.PARQUET.value}")
        if output_dir is not None
        else None
    )

    intersection_dataframe = read_intersection_dataframes(
        intersected_featuremaps_parquet,
        snp_error_rate=snp_error_rate_hdf,
        output_parquet=intersection_dataframe_fname,
    )
    signature_dataframe = read_signature(
        signature_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        sample_name=sample_name,
    )
    return signature_dataframe, intersection_dataframe
