from __future__ import annotations

import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from os.path import basename, dirname
from os.path import join as pjoin
from os.path import splitext

import numpy as np
import pandas as pd
import pyBigWig as bw
import pysam
from joblib import Parallel, delayed
from tqdm import tqdm

from ugvc import logger
from ugvc.dna.utils import revcomp
from ugvc.utils.consts import FileExtension
from ugvc.vcfbed.variant_annotation import get_trinuc_substitution_dist, parse_trinuc_sub

default_featuremap_info_fields = {
    "X_CIGAR": str,
    "X_EDIST": int,
    "X_FC1": str,
    "X_FC2": str,
    "X_READ_COUNT": int,
    "X_FILTERED_COUNT": int,
    "X_FLAGS": int,
    "X_LENGTH": int,
    "X_MAPQ": float,
    "X_INDEX": int,
    "X_RN": str,
    "X_SCORE": float,
    "rq": float,
}


class FeaturemapAnnotator(VcfAnnotator):
    """
    Annotate vcf with featuremap-specific fields derived from X_FLAGS and X_CIGAR:
    - is_forward: is the read forward mapped
    - is_duplicate: is the read a duplicate
    - max_softclip_length: maximum softclip length
    """

    IS_FORWARD = "is_forward"
    IS_DUPLICATE = "is_duplicate"
    MAX_SOFTCLIP_LENGTH = "max_softclip_length"
    X_FLAGS = "X_FLAGS"
    X_CIGAR = "X_CIGAR"

    def __init__(self):
        pass

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to include new fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            Modified VCF header

        """
        header.add_line("##ugvc_ugvc/mrd/mrd_utils.py_FeaturemapAnnotator=.")
        header.add_line(
            f"##INFO=<ID={FeaturemapAnnotator.IS_FORWARD},"
            'Number=1,Type=Flag,Description="is the read forward mapped">'
        )
        header.add_line(
            f"##INFO=<ID={FeaturemapAnnotator.IS_DUPLICATE},"
            'Number=1,Type=Flag,Description="is the read a duplicate">'
        )
        header.add_line(
            f"##INFO=<ID={FeaturemapAnnotator.MAX_SOFTCLIP_LENGTH},"
            'Number=1,Type=Integer,Description="maximum softclip length">'
        )

        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of updated VCF records
        """
        records_out = [None] * len(records)
        for j, record in enumerate(records):
            if FeaturemapAnnotator.X_FLAGS in record.info:
                flags = record.info[FeaturemapAnnotator.X_FLAGS]
                record.info[FeaturemapAnnotator.IS_FORWARD] = (flags & 16) == 0
                record.info[FeaturemapAnnotator.IS_DUPLICATE] = (flags & 1024) != 0
            if FeaturemapAnnotator.X_CIGAR in record.info:
                record.info[FeaturemapAnnotator.MAX_SOFTCLIP_LENGTH] = get_max_softclip_len(
                    record.info[FeaturemapAnnotator.X_CIGAR]
                )
            records_out[j] = record

        return records_out


def _get_sample_name_from_file_name(file_name, split_position=0):
    """
    Internal formatting of filename for mrd pipeline

    Parameters
    ----------
    file_name: str
        file name
    split_position: int
        which position relative to splitting by "." the sample name is

    Returns
    -------
    out_file_name
        reformatted file name

    """
    return (
        basename(file_name)
        .split(".")[split_position]
        .replace("-", "_")
        .replace("_filtered_signature", "")  # remove generic suffixes
        .replace("_signature", "")  # remove generic suffixes
        .replace("signature_", "")  # remove generic suffixes
        .replace("_filtered", "")  # remove generic suffixes
        .replace("filtered", "")  # remove generic suffixes
        .replace("featuremap_", "")  # remove generic suffixes
        .replace("_featuremap", "")  # remove generic suffixes
        .replace("featuremap", "")  # remove generic suffixes
    )


def _get_hmer_length(ref: str, left_motif: str, right_motif: str):
    """
    Calculate the length of the hmer the ref is contained in (limited by input motif length)
    Parameters
    ----------
    ref: str
        reference base (single base)
    left_motif: str
        left motif in forward orientation
    right_motif: str
        right motif in forward orientation

    Returns
    -------
    hlen: int
        The homopolymer length the reference base is contained in (limited by input motif length)

    """

    left_motif_len = len(left_motif)
    x = np.array([(k == ref, sum(1 for _ in v)) for k, v in itertools.groupby(left_motif + ref + right_motif)])
    return x[np.argmax(x[:, 1].cumsum() >= left_motif_len + 1), 1]


def _safe_tabix_index(vcf_file: str):
    try:
        pysam.tabix_index(vcf_file, preset="vcf", force=True)  # make sure vcf is indexed
        if (not os.path.isfile(vcf_file)) and os.path.isfile(vcf_file + ".gz"):  # was compressed by tabix
            vcf_file = vcf_file + ".gz"
    except Exception as e:  # pylint: disable=broad-except
        # Catching a broad exception because this is optional and we never want to have a run fail because of the index
        logger.warning(f"Could not index signature file {vcf_file}\n{str(e)}")
    return vcf_file


def _validate_info(x):
    """
    Validate INFO field in a VCF record.
    In some cases it is a string of boolean ("TRUE"/"FALSE")
    In the case of long hmers it's numeric: if the hmer >= 7
    it appears in the INFO field as a tuple of (hmer, hmer_length),
    then the function should return True
    """
    MAX_HMER_LENGTH = 7
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        if x.lower() == "true":
            return True
        if x.lower() == "false":
            return False
        if x.isnumeric():
            return bool(float(x) >= MAX_HMER_LENGTH)
        raise ValueError(f"Cannot convert {x} to bool")
    return None

def read_signature(  # pylint: disable=too-many-branches,too-many-arguments
    signature_vcf_files: list[str],
    output_parquet: str = None,
    coverage_bw_files: list[str] = None,
    tumor_sample: str = None,
    x_columns_name_dict: dict = None,
    columns_to_drop: list = None,
    verbose: bool = True,
    raise_exception_on_sample_not_found: bool = False,
    signature_type: str = None,
    return_dataframes: bool = False,
    concat_to_existing_output_parquet: bool = False,
):
    """
    Read signature (variant calling output, generally mutect) results to dataframe.

    signature_vcf_files: str or list[str]
        File name or a list of file names
    output_parquet: str, optional
        File name to save result to, unless None (default).
        If this file exists and concat_to_existing_output_parquet is True data is apppended
    coverage_bw_files: list[str], optional
        Coverage bigwig files generated with "coverage_analysis full_analysis", defualt None
    tumor_sample: str, optional
        tumor sample name in the vcf to take allele fraction (AF) from. If not given then a line starting with
        '##tumor_sample=' is looked for in the header, and if it's not found sample data is not read.
    x_columns_name_dict: dict, optional
        Dictionary of INFO fields starting with "X-" (custom UG fields) to read as keys and respective columns names
        as values, if None (default) this value if used:
        {"X-CSS": "cycle_skip_status","X-GCC": "gc_content","X-LM": "left_motif","X-RM": "right_motif"}
    columns_to_drop: list, optional
        List of columns to drop, as values, if None (default) this value if used: ["X-IC"]
    verbose: bool, optional
        show verbose debug messages
    raise_exception_on_sample_not_found: bool, optional
        if True (default False) and the sample name could not be found to determine AF, raise a ValueError
    signature_type: str, optional
        Set signature_type column in the dataframe to be equal to the input value, if it is not None
    concat_to_existing_output_parquet: bool, optional
        If True (default False) and output_parquet is not None and it exists, the new data is concatenated to the


    Raises
    ------
    ValueError
        may be raised

    """
    if output_parquet and os.path.isfile(output_parquet):  # concat new data with existing
        if not concat_to_existing_output_parquet:
            raise ValueError(
                f"output_parquet {output_parquet} exists and concat_to_existing_output_parquet is False"
                " - cannot continue"
            )
        logger.debug(f"Reading existing data from {output_parquet}")
        df_existing = pd.read_parquet(output_parquet)
    else:
        df_existing = None

    if output_parquet is None and not return_dataframes:
        raise ValueError("output_parquet is not None and return_dataframes is False - nothing to do")

    if signature_vcf_files is None or len(signature_vcf_files) == 0:
        logger.debug("Empty input to read_signature - exiting without doing anything")
        return df_existing
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    if not isinstance(signature_vcf_files, str) and isinstance(signature_vcf_files, Iterable):
        logger.debug(f"Reading and merging signature files:\n {signature_vcf_files}")
        df_sig = pd.concat(
            (
                read_signature(
                    file_name,
                    output_parquet=None,
                    return_dataframes=True,
                    tumor_sample=tumor_sample,
                    x_columns_name_dict=x_columns_name_dict,
                    columns_to_drop=columns_to_drop,
                    verbose=j == 0,  # only verbose in first iteration
                )
                .assign(signature=_get_sample_name_from_file_name(file_name, split_position=0))
                .reset_index()
                .set_index(["signature", "chrom", "pos"])
                for j, file_name in enumerate(np.unique(signature_vcf_files))
            )
        )

    else:
        entries = []
        logger.debug(f"Reading vcf file {signature_vcf_files}")
        x_columns_name_dict = {
            "X_CSS": "cycle_skip_status",
            "X_GCC": "gc_content",
            "X_LM": "left_motif",
            "X_RM": "right_motif",
        }
        columns_to_drop = ["X_IC", "X_HIL", "X_HIN", "X_IC", "X_IL"]

        signature_vcf_files = _safe_tabix_index(signature_vcf_files)  # make sure it's indexed
        with pysam.VariantFile(signature_vcf_files) as variant_file:
            info_keys = list(variant_file.header.info.keys())
            # get tumor sample name
            header = variant_file.header
            if tumor_sample is None:
                number_of_samples = len(list(header.samples))
                if number_of_samples >= 2:
                    # mutect2 vcf
                    for x in str(header).split("\n"):
                        m = re.match(r"##tumor_sample=(.+)", x)  # good for mutect2
                        if m is not None:
                            tumor_sample = m.groups()[0]
                            logger.debug(f"Tumor sample name is {tumor_sample}")
                            break
                elif number_of_samples == 1:
                    # DV vcf
                    tumor_sample = list(header.samples)[0]
                    logger.debug(f"Tumor sample name is {tumor_sample}")
                elif number_of_samples == 0:
                    # synthetic signature
                    logger.debug("No samples found in vcf file")
            if tumor_sample is not None and tumor_sample not in header.samples:
                if raise_exception_on_sample_not_found:
                    raise ValueError(
                        f"Tumor sample name {tumor_sample} not in vcf sample names: {list(header.samples)}"
                    )
                tumor_sample = None
            if tumor_sample is None and raise_exception_on_sample_not_found:
                raise ValueError(f"Tumor sample name not found. vcf sample names: {list(header.samples)}")
            # not found but allowed to continue because raise_exception_on_sample_not_found = False
            # get INFO annotations and X_ variables
            genomic_region_annotations = [
                k for k in info_keys if variant_file.header.info[k].description.startswith("Genomic Region Annotation:")
            ]
            x_columns = [k for k in info_keys if k.startswith("X_") and k not in columns_to_drop]
            x_columns_name_dict = {k: x_columns_name_dict.get(k, k) for k in x_columns}
            format_fields = [x.name for x in header.formats.values()]
            if "AF" in format_fields:
                # mutect2 vcf
                af_field = "AF"
            elif "VAF" in format_fields:
                # DV vcf
                af_field = "VAF"
            elif "DNA_VAF" in info_keys:
                # synthetic signature
                af_field = "DNA_VAF"
            else:
                af_field = None
            logger.debug(f"Reading x_columns: {x_columns}")

            for j, rec in enumerate(variant_file):
                if j == 0 and tumor_sample is None:
                    samples = rec.samples.keys()
                    if len(samples) == 1:
                        tumor_sample = samples[0]
                entries.append(
                    tuple(
                        [
                            rec.chrom,
                            rec.pos,
                            rec.ref,
                            rec.alts[0],
                            rec.id,
                            rec.qual,
                            (
                                rec.samples[tumor_sample][af_field][0]
                                if tumor_sample
                                and af_field in rec.samples[tumor_sample]
                                and len(rec.samples[tumor_sample]) > 0
                                else rec.info[af_field]
                                if af_field in rec.info
                                else np.nan
                            ),
                            (
                                rec.samples[tumor_sample]["VAF"][0]
                                if tumor_sample and "VAF" in rec.samples[tumor_sample]
                                else np.nan
                            ),
                            (
                                rec.samples[tumor_sample]["DP"]
                                if tumor_sample and "DP" in rec.samples[tumor_sample]
                                else np.nan
                            ),
                            rec.info["LONG_HMER"] if "LONG_HMER" in rec.info else np.nan,
                            rec.info["TLOD"][0]
                            if "TLOD" in rec.info and isinstance(rec.info["TLOD"], Iterable)
                            else np.nan,
                            rec.info["SOR"] if "SOR" in rec.info else np.nan,
                        ]
                        + [_validate_info(rec.info.get(c, "False")) for c in genomic_region_annotations]
                        + [rec.info[c][0] if isinstance(rec.info[c], tuple) else rec.info[c] for c in x_columns]
                    )
                )
        logger.debug(f"Done reading vcf file {signature_vcf_files}")
        logger.debug("Converting to dataframe")
        df_sig = (
            pd.DataFrame(
                entries,
                columns=[
                    "chrom",
                    "pos",
                    "ref",
                    "alt",
                    "id",
                    "qual",
                    "af",
                    "vaf",
                    "depth_tumor_sample",
                    "hmer",
                    "tlod",
                    "sor",
                ]
                + genomic_region_annotations
                + [x_columns_name_dict[c] for c in x_columns],
            )
            .reset_index(drop=True)
            .astype({"chrom": str, "pos": int})
            .set_index(["chrom", "pos"])
            .dropna(how="all", axis=1)
        )
        logger.debug("Done converting to dataframe")

    df_sig = df_sig.sort_index()

    if coverage_bw_files:  # collect coverage per locus
        try:
            logger.debug("Reading input from bigwig coverage data")
            f_bw = [bw.open(x) for x in coverage_bw_files]
            df_list = []

            for chrom, df_tmp in tqdm(df_sig.groupby(level="chrom")):
                if df_tmp.shape[0] == 0:
                    continue
                found_correct_file = False

                f_bw_chrom = {}
                for f_bw_chrom in f_bw:
                    if chrom in f_bw_chrom.chroms():
                        found_correct_file = True
                        break

                if not found_correct_file:
                    raise ValueError(f"Could not find a bigwig file with {chrom} in:\n{', '.join(coverage_bw_files)}")

                chrom_start = df_tmp.index.get_level_values("pos") - 1
                chrom_end = df_tmp.index.get_level_values("pos")
                df_list.append(
                    df_tmp.assign(
                        coverage=np.concatenate(
                            [f_bw_chrom.values(chrom, x, y, numpy=True) for x, y in zip(chrom_start, chrom_end)]
                        )
                    )
                )
        finally:
            if "f_bw" in locals():
                for variant_file in f_bw:
                    variant_file.close()
        df_sig = pd.concat(df_list).astype({"coverage": int})
    df_sig["hmer"] = pd.to_numeric(df_sig["hmer"], errors="coerce")
    logger.debug("Calculating reference hmer")
    try:
        df_sig.loc[:, "hmer"] = (
            df_sig[["hmer"]]
            .assign(
                hmer_calc=df_sig.apply(
                    lambda row: _get_hmer_length(row["ref"], row["left_motif"], row["right_motif"]),
                    axis=1,
                )
            )
            .astype(float)
            .max(axis=1)
        )
    except KeyError:
        logger.debug("Could not calculate hmer")

    logger.debug("Annotating with mutation type (ref->alt)")
    ref_is_c_or_t = df_sig["ref"].isin(["C", "T"])
    df_sig.loc[:, "mutation_type"] = (
        np.where(ref_is_c_or_t, df_sig["ref"], df_sig["ref"].apply(revcomp))
        + "->"
        + np.where(ref_is_c_or_t, df_sig["alt"], df_sig["alt"].apply(revcomp))
    )

    df_sig.columns = [x.replace("-", "_").lower() for x in df_sig.columns]

    if signature_type is not None and isinstance(signature_type, str):
        df_sig = df_sig.assign(signature_type=signature_type)

    if df_existing is not None:
        logger.debug(f"Concatenating to previous existing data in {output_parquet}")
        df_sig = pd.concat((df_existing.set_index(df_sig.index.names), df_sig))

    if output_parquet:
        logger.debug(f"Saving output signature/s to {output_parquet}")
        df_sig.reset_index().to_parquet(output_parquet)

    if return_dataframes:
        return df_sig
    return None


def read_intersection_dataframes(
    intersected_featuremaps_parquet,
    output_parquet=None,
    return_dataframes=False,
):
    """
    Read featuremap dataframes from several intersections of one featuremaps with several signatures, each is annotated
    with the signature name. Assumed to be the output of featuremap.intersect_featuremap_with_signature
    Return a concatenated dataframe with the signature name as a column

    Parameters
    ----------
    intersected_featuremaps_parquet: list[str] or str
        list of featuremaps intesected with various signatures
    output_parquet: str
        File name to save result to, default None
    return_dataframes: bool
        Return dataframes

    Returns
    -------
    dataframe: pd.DataFrame
        concatenated intersection dataframes

    Raises
    ------
    ValueError
        may be raised

    """
    if output_parquet is None and not return_dataframes:
        raise ValueError("output_parquet is not None and return_dataframes is False - nothing to do")
    if isinstance(intersected_featuremaps_parquet, str):
        intersected_featuremaps_parquet = [intersected_featuremaps_parquet]
    logger.debug(f"Reading {len(intersected_featuremaps_parquet)} intersection featuremaps")
    df_int = pd.concat(
        (
            pd.read_parquet(f).assign(
                cfdna=_get_sample_name_from_file_name(f, split_position=0),
                signature=_get_sample_name_from_file_name(f, split_position=1),
                signature_type=_get_sample_name_from_file_name(f, split_position=2),
            )
            .reset_index()
            .astype(
                {
                    "chrom": str,
                    "pos": int,
                    "ref": str,
                    "alt": str,
                    "left_motif": str,
                    "right_motif": str,
                }
            )
            .set_index(["signature", "chrom", "pos"])
            for f in intersected_featuremaps_parquet
        )
    )
    if output_parquet is not None:
        df_int.reset_index().to_parquet(output_parquet)
    if return_dataframes:
        return df_int
    return None


def intersect_featuremap_with_signature(
    featuremap_file: str,
    signature_file: str,
    output_intersection_file: str = None,
    signature_type: str = None,
    add_info_to_header: bool = True,
    overwrite: bool = True,
) -> str:
    """
    Intersect featuremap and signature vcf files on chrom, position, ref and alts (require same alts), keeping all the
    entries in featuremap. Lines from featuremap propagated to output

    Parameters
    ----------
    featuremap_file: str
        Of cfDNA
    signature_file: str
        VCF file, tumor variant calling results
    output_intersection_file: str, optional
        Output vcf file, .vcf.gz or .vcf extension, if None (default) determined automatically from file names
    signature_type: str, optional
        If defined and tag the file name accordingly as "*.matched.vcf.gz", "*.control.vcf.gz" or "*.db_control.vcf.gz"
    add_info_to_header: bool, optional
        Add line to header to indicate this function ran (default True)
    overwrite: bool, optional
        Force rewrite of output (if false and output file exists an OSError will be raised). Default True.

    Returns
    -------
    output_intersection_file: str
        Output vcf file, either identical to input or automatically determined from input file names

    Raises
    ------
    OSError
        in case the file already exists and function executed with no overwrite=True
    ValueError
        If input output_intersection_file does not end with .vcf or .vcf.gz

    """
    # parse the file name
    if output_intersection_file is None:
        featuremap_name = _get_sample_name_from_file_name(featuremap_file, split_position=0)
        signature_name = _get_sample_name_from_file_name(signature_file, split_position=0)
        if signature_type is None:
            type_name = ""
        else:
            type_name = f".{signature_type}"
        output_intersection_file = f"{featuremap_name}.{signature_name}{type_name}.intersection.vcf.gz"
        logger.debug(f"Output file name will be: {output_intersection_file}")
    if not (output_intersection_file.endswith(".vcf.gz") or output_intersection_file.endswith(".vcf")):
        raise ValueError(f"Output file must end with .vcf or .vcf.gz, got {output_intersection_file}")
    output_intersection_file_vcf = (
        output_intersection_file[:-3] if output_intersection_file.endswith(".gz") else output_intersection_file
    )
    if (not overwrite) and (os.path.isfile(output_intersection_file) or os.path.isfile(output_intersection_file_vcf)):
        raise OSError(f"Output file {output_intersection_file} already exists and overwrite flag set to False")
    # make sure vcf is indexed
    signature_file = _safe_tabix_index(signature_file)

    with tempfile.TemporaryDirectory(dir=dirname(output_intersection_file)) as temp_dir:
        isec_file = os.path.join(temp_dir, "isec.vcf.gz")
        header_file = os.path.join(temp_dir, "header.txt")
        headerless_featuremap = os.path.join(temp_dir, "headerless_featuremap.vcf.gz")
        featuremap_isec_by_pos = os.path.join(temp_dir, "featuremap_isec_by_pos.vcf.gz")

        # Extract the header from featuremap_file and write to a new file
        cmd = f"bcftools view -h {featuremap_file} | head -n-1 - > {header_file}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)

        # Add comment lines to the header
        with open(header_file, "a", encoding="utf-8") as f:
            if add_info_to_header:
                f.write(
                    "##File:Description=This file is an intersection "
                    "of a featuremap with a somatic mutation signature\n"
                )
                f.write(f"##python_cmd:intersect_featuremap_with_signature=python {' '.join(sys.argv)}\n")
                if signature_type == "matched":
                    f.write("##Intersection:type=matched (signature and featuremap from the same patient)\n")
                else:
                    f.write("##Intersection:type=control (signature and featuremap not from the same patient)\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # use bedtools intersect
        cmd = f"bedtools intersect -a {featuremap_file} -b {signature_file} -wa | \
            cat {header_file} - | \
            bgzip > {featuremap_isec_by_pos} && bcftools index -t {featuremap_isec_by_pos}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)

        # Use bcftools isec to intersect the two VCF files by position and compress the output
        cmd = f"bcftools isec -n=2 -w1 -Oz -o {isec_file} {signature_file} {featuremap_isec_by_pos}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)

        cmd = f"bcftools view -H {featuremap_isec_by_pos} > {headerless_featuremap}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)

        # Use awk to filter the intersected file for records with matching alt alleles and compress the output
        # this awk line matches the first, second, fourth and fifth columns of two files and prints the matched
        # line from file2 for the first file, it creates an associative array 'a'. if the same array appears in
        # the second file, it prints the line of the second file
        awk_part = "awk 'NR==FNR{a[$1,$2,$4,$5];next} ($1,$2,$4,$5) in a'"
        cmd = f"bcftools view {isec_file} | \
            {awk_part} - {headerless_featuremap} | \
            cat {header_file} - | \
            bcftools view - -Oz -o {output_intersection_file} && bcftools index -t {output_intersection_file}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)

        # Assert output
        assert os.path.isfile(output_intersection_file)
    return output_intersection_file


def vcf_row_generator(
    vcf_handle: pysam.VariantFile,
    info_metadata_dict: dict = None,
    formats_metadata_dict: dict = None,
    sample_index: int = None,
):
    """
    Return a generator object that reads entries from an open vcf file handle and yields a list of values

    Parameters
    ----------
    vcf_handle: pysam.VariantFile
        An open vcf file handle
    info_metadata_dict: dict
        A dictionary of info columns to read from the vcf file. The keys are the info column names and the values
        default None, in which case no INFO fields are read
    formats_metadata_dict: dict
        A dictionary of format columns to read from the vcf file. The keys are the info column names and the values
        default None, in which case no FORMAT fields are read
    sample_index: int
        The index of the sample to read from the vcf file.

    """
    for record in vcf_handle.fetch():
        # read basic fields
        entry = [
            record.chrom,
            record.pos,
            record.ref,
            record.alts[0] if len(record.alts) >= 1 else None,
            record.qual,
            ";".join(record.filter.keys()),
        ]
        # read info fields
        if info_metadata_dict is not None and len(info_metadata_dict) > 0:
            entry += [
                record.info.get(info_key)[0]
                if info_value[0] == "A"
                and record.info.get(info_key, None) is not None
                and len(record.info.get(info_key)) >= 1
                else record.info.get(info_key, np.nan)
                for info_key, info_value in info_metadata_dict.items()
            ]
        # read format fields
        if formats_metadata_dict is not None and len(formats_metadata_dict) > 0:
            entry += [
                record.samples[sample_index].get(formats_key, np.nan) for formats_key in formats_metadata_dict.keys()
            ]
        yield entry


def vcf_row_generator(
    vcf_handle: pysam.VariantFile,
    info_metadata_dict: dict = None,
    formats_metadata_dict: dict = None,
    sample_index: int = None,
):
    """
    Return a generator object that reads entries from an open vcf file handle and yields a list of values

    Parameters
    ----------
    vcf_handle: pysam.VariantFile
        An open vcf file handle
    info_metadata_dict: dict
        A dictionary of info columns to read from the vcf file. The keys are the info column names and the values
        default None, in which case no INFO fields are read
    formats_metadata_dict: dict
        A dictionary of format columns to read from the vcf file. The keys are the info column names and the values
        default None, in which case no FORMAT fields are read
    sample_index: int
        The index of the sample to read from the vcf file.

    """
    for record in vcf_handle.fetch():
        # read basic fields
        entry = [
            record.chrom,
            record.pos,
            record.ref,
            record.alts[0] if len(record.alts) >= 1 else None,
            record.qual,
            ";".join(record.filter.keys()),
        ]
        # read info fields
        if info_metadata_dict is not None and len(info_metadata_dict) > 0:
            entry += [
                record.info.get(info_key)[0]
                if info_value[0] == "A"
                and record.info.get(info_key, None) is not None
                and len(record.info.get(info_key)) >= 1
                else record.info.get(info_key, np.nan)
                for info_key, info_value in info_metadata_dict.items()
            ]
        # read format fields
        if formats_metadata_dict is not None and len(formats_metadata_dict) > 0:
            entry += [
                record.samples[sample_index].get(formats_key, np.nan) for formats_key in formats_metadata_dict.keys()
            ]
        yield entry


# pylint: disable=too-many-arguments
def featuremap_to_dataframe(
    featuremap_vcf: str,
    output_file: str = None,
    input_info_fields: list[str] = "all",
    input_format_fields: list[str] = None,
    sample_name: str = None,
    sample_index: int = None,
    default_int_fillna_value: int = 0,
    default_string_fillna_value: int = np.nan,
):
    """
    Converts featuremap in vcf format to dataframe

    Parameters
    ----------
    featuremap_vcf : str
        Path to featuremap vcf file
    output_file : str, optional
        Path to output file, by default None
        If None, saved to the same basename as featuremap_vcf with .parquet extension
    input_info_fields : list[str], optional
        List of input info fields to include in dataframe, by default 'all''
        If 'all' then all the info fields are read to columns
        If None then no info fields are read to columns
    input_format_fields : list[str], optional
        List of format info fields to include in dataframe, by default None
        If 'all' then all the info fields are read to columns
        If None then no info fields are read to columns
    sample_name : str, optional
        Name of sample to read formats columns from, default None
        If sample_name and sample_index are both None then no formats columns are read
    sample_index : int, optional
        Index of sample to read formats columns from, default None
        If sample_name and sample_index are both None and format_info_fields is not empty
        then sample_index defaults to 0
    default_int_fillna_value : int, optional
        Value to fill na values with for Integer fields, by default 0
    default_string_fillna_value : str, optional
        Value to fill na values with for String fields, by default np.nan

    Returns
    -------
    pd.DataFrame
        Dataframe of featuremap

    Raises
    ------
    ValueError
        If sample_name and sample_index are both not None
        If sample_name is not in vcf file
        If sample_index is out of range
        If input_info_fields contains a field not in vcf header
        If format_info_fields contains a field not in vcf header

    """
    # check inputs
    if sample_name is not None and sample_index is not None:
        raise ValueError("Cannot specify both sample_name and sample_index")
    if sample_name is not None and sample_name not in pysam.VariantFile(featuremap_vcf).header.samples:
        raise ValueError(f"Sample {sample_name} not in vcf file")
    if sample_index is not None and sample_index >= len(pysam.VariantFile(featuremap_vcf).header.samples):
        raise ValueError(f"Sample index {sample_index} out of range")
    if input_info_fields is not None and input_info_fields != "all" and len(input_info_fields) > 0:
        for info_field in input_info_fields:
            if info_field not in pysam.VariantFile(featuremap_vcf).header.info:
                raise ValueError(f"Input info field {info_field} not in vcf header")
    if input_format_fields is not None and input_format_fields != "all" and len(input_format_fields) > 0:
        for info_field in input_format_fields:
            if info_field not in pysam.VariantFile(featuremap_vcf).header.formats:
                raise ValueError(f"Format info field {info_field} not in vcf header")
        if sample_index is None and sample_name is None:
            sample_index = 0
        if sample_index is None and sample_name is not None:
            sample_index = list(pysam.VariantFile(featuremap_vcf).header.samples).index(sample_name)

    # define type conversion dictionary, String is converted to object to support np.nan
    type_conversion_dict = {"String": object, "Integer": int, "Float": float, "Flag": bool}

    # auxiliary function to parse info and formats fields to read
    def get_metadata_dict(values: list[str], input_fields: list[str] = None):
        if input_fields is None:
            return {}, []
        metadata_dict_full = {value.name: (value.number, value.type) for value in values}
        input_fields_to_use = list(metadata_dict_full.keys()) if input_fields.lower() == "all" else input_fields
        for key in input_fields_to_use:
            if key not in metadata_dict_full:
                raise ValueError(f"Input field {key} missing from vcf header")
        metadata_dict = {key: metadata_dict_full[key] for key in input_fields_to_use}
        return metadata_dict, input_fields_to_use

    # read vcf file to dataframe
    with pysam.VariantFile(featuremap_vcf) as vcf_handle:
        # parse header and get metadata
        header = vcf_handle.header
        info_metadata_dict, info_input_fields_to_use = get_metadata_dict(
            values=header.info.values(), input_fields=input_info_fields
        )
        format_metadata_dict, format_input_fields_to_use = get_metadata_dict(
            values=header.formats.values(), input_fields=input_format_fields
        )
        # This section is a patch for specific integer tags, since there is an issue with FeautreMap
        # not propagating the types of copied SAM tags properly, so that they all default to String.
        # There is a pending request to fix, in the mean time this is a workaround.
        harcoded_info_fields = {x: "Integer" for x in ("a3", "ae", "as", "s2", "s3", "te", "ts")}
        harcoded_info_fields["rq"] = "Float"
        for field, datatype in harcoded_info_fields.items():
            if field in info_metadata_dict:
                info_metadata_dict[field] = (info_metadata_dict[field][0], datatype)
        # read vcf file to dataframe
        df = pd.DataFrame(
            vcf_row_generator(
                vcf_handle=vcf_handle,
                info_metadata_dict=info_metadata_dict,
                formats_metadata_dict=format_metadata_dict,
                sample_index=sample_index,
            ),
            columns=["chrom", "pos", "ref", "alt", "qual", "filter"]
            + info_input_fields_to_use
            + format_input_fields_to_use,
        )
    # fillna in int columns before converting types, because int columns can't contain NaNs
    df = df.fillna(
        {
            info_key: default_int_fillna_value
            for info_key in info_metadata_dict.keys()
            if type_conversion_dict[info_metadata_dict[info_key][1]] == int
        }
    )
    df.fillna(
        {
            info_key: default_string_fillna_value
            for info_key in info_metadata_dict.keys()
            if type_conversion_dict[info_metadata_dict[info_key][1]] == object
        }
    )
    # convert types
    df = df.astype(
        {
            **{"qual": float},
            **{
                info_key: type_conversion_dict[info_metadata_dict[info_key][1]]
                for info_key in info_metadata_dict.keys()
                if info_metadata_dict[info_key][0]
            },
        }
    )
    # Determine output file name
    if output_file is None:
        if featuremap_vcf.endswith(FileExtension.VCF_GZ.value):
            output_file = pjoin(
                dirname(featuremap_vcf),
                splitext(splitext(basename(featuremap_vcf))[0])[0] + FileExtension.PARQUET.value,
            )
        else:
            output_file = pjoin(
                dirname(featuremap_vcf),
                splitext(basename(featuremap_vcf))[0] + FileExtension.PARQUET.value,
            )
    elif not output_file.endswith(FileExtension.PARQUET.value):
        output_file = f"{output_file}{FileExtension.PARQUET.value}"

    # Save and return
    df.to_parquet(output_file)
    return df


def prepare_data_from_mrd_pipeline(
    intersected_featuremaps_parquet,
    matched_signatures_vcf_files=None,
    control_signatures_vcf_files=None,
    db_control_signatures_vcf_files=None,
    coverage_bw_files=None,
    tumor_sample=None,
    output_dir=None,
    output_basename=None,
    return_dataframes=False,
):
    """

    intersected_featuremaps_parquet: list[str]
        list of featuremaps intesected with various signatures
    matched_signatures_vcf_files: list[str]
        File name or a list of file names, signature vcf files of matched signature/s
    control_signatures_vcf_files: list[str]
        File name or a list of file names, signature vcf files of control signature/s
    db_control_signatures_vcf_files: list[str]
        File name or a list of file names, signature vcf files of db (synthetic) control signature/s
    coverage_bw_files: list[str]
        Coverage bigwig files generated with "coverage_analysis full_analysis", disabled (None) by default
    tumor_sample: str
        sample name in the vcf to take allele fraction (AF) from.
    output_dir: str
        path to which output will be written if not None (default None)
    output_basename: str
        basename of output file (if output_dir is not None must also be not None), default None

    Returns
    -------
    dataframe: pd.DataFrame
        merged data for MRD analysis

    Raises
    -------
    OSError
        in case the file already exists and function executed with no force overwrite
    ValueError
        may be raised
    """
    if output_dir is not None and output_basename is None:
        raise ValueError(f"output_dir is not None ({output_dir}) but output_basename is")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    if matched_signatures_vcf_files is None:
        raise ValueError("matched_signatures_vcf_files must be defined")
    if control_signatures_vcf_files is None and db_control_signatures_vcf_files is None:
        raise ValueError("Either control_signatures_vcf_files or db_control_signatures_vcf_files must be defined")

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
        output_parquet=intersection_dataframe_fname,
    )
    signature_dataframe = read_signature(
        matched_signatures_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        tumor_sample=tumor_sample,
        signature_type="matched",
        return_dataframes=return_dataframes,
        concat_to_existing_output_parquet=False,
    )
    signature_dataframe = read_signature(
        control_signatures_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        tumor_sample=tumor_sample,
        signature_type="control",
    )
    signature_dataframe = read_signature(
        db_control_signatures_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        tumor_sample=tumor_sample,
        signature_type="db_control",
        return_dataframes=return_dataframes,
        concat_to_existing_output_parquet=True,
    )

    intersection_dataframe = read_intersection_dataframes(
        intersected_featuremaps_parquet,
        substitution_error_rate=substitution_error_rate_hdf,
        output_parquet=intersection_dataframe_fname,
        return_dataframes=return_dataframes,
    )

    if return_dataframes:
        return signature_dataframe, intersection_dataframe
    return None


def concat_dataframes(dataframes: list, outfile: str, n_jobs: int = 1):
    df = pd.concat(Parallel(n_jobs=n_jobs)(delayed(pd.read_parquet)(f) for f in dataframes))
    df = df.sort_index()
    if isinstance(df.index, pd.MultiIndex):  # DataFrames are saved as not multi-indexed so parquet doesn't break
        df = df.reset_index()
    df.to_parquet(outfile)
    return df


def generate_synthetic_signatures(
    signature_vcf: str, db_vcf: str, n_synthetic_signatures: int, output_dir: str
) -> list[str]:
    """
    Generate synthetic signatures from a signature vcf file and a db vcf file

    Parameters
    ----------
    signature_vcf: str
        signature vcf file
    db_vcf: str
        db vcf file
    n_synthetic_signatures: int
        number of synthetic signatures to generate
    output_dir: str
        output directory

    Returns
    -------
    synthetic_signatures: list[str]
        list of synthetic control vcf files

    """
    # parse trinuc substitution distribution from signature vcf and db vcf
    trinuc_signature = get_trinuc_substitution_dist(signature_vcf)
    trinuc_db = get_trinuc_substitution_dist(db_vcf)

    # allocate index and counter per synthetic signature
    trinuc_dict = {}
    # fix a seed for random.choice
    random_seed: int = 0
    np.random.seed(random_seed)
    for trinucsub in trinuc_signature.keys():
        n_subs_signature = trinuc_signature[trinucsub]
        n_subs_db = trinuc_db[trinucsub]
        trinuc_dict[trinucsub] = {}
        trinuc_dict[trinucsub]["index_to_sample"] = {}
        trinuc_dict[trinucsub]["trinuc_counter"] = {}
        for i in range(n_synthetic_signatures):
            trinuc_dict[trinucsub]["index_to_sample"][i] = np.random.choice(
                range(0, n_subs_db), n_subs_signature, replace=False
            )
            trinuc_dict[trinucsub]["trinuc_counter"][i] = 0

    # make output directory
    os.makedirs(output_dir, exist_ok=True)
    # db and signautre basename for output files
    db_basename = splitext(splitext(basename(db_vcf))[0])[0]
    signature_basename = splitext(splitext(basename(signature_vcf))[0])[0]
    # read db vcf file and allocate output files and file handles
    db_fh = pysam.VariantFile(db_vcf, "rb")
    synthetic_signatures = []
    file_handle_list = []
    for i in range(n_synthetic_signatures):
        output_vcf = pjoin(output_dir, f"syn{i}_{signature_basename}_{db_basename}.vcf.gz")
        synthetic_signatures.append(output_vcf)
        outfh = pysam.VariantFile(output_vcf, "w", header=db_fh.header)
        file_handle_list.append(outfh)

    # read db file, parse trinuc substiution and write to output files
    for rec in db_fh.fetch():
        db_trinucsub = parse_trinuc_sub(rec)
        for i in range(n_synthetic_signatures):
            if trinuc_dict[db_trinucsub]["trinuc_counter"][i] in trinuc_dict[db_trinucsub]["index_to_sample"][i]:
                file_handle_list[i].write(rec)
            trinuc_dict[db_trinucsub]["trinuc_counter"][i] += 1

    db_fh.close()

    for i in range(n_synthetic_signatures):
        file_handle_list[i].close()
        output_vcf = synthetic_signatures[i]
        pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)

    return synthetic_signatures
