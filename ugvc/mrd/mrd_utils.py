from __future__ import annotations

import itertools
import logging
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from os.path import basename
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pyBigWig as bw
import pysam
from joblib import Parallel, delayed
from tqdm import tqdm

from ugvc import logger
from ugvc.dna.format import CHROM_DTYPE, DEFAULT_FLOW_ORDER
from ugvc.dna.utils import get_max_softclip_len, revcomp
from ugvc.utils.consts import FileExtension
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe, get_motif_around

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


def read_signature(  # pylint: disable=too-many-branches,too-many-arguments
    signature_vcf_files: list[str],
    output_parquet: str = None,
    coverage_bw_files: list[str] = None,
    tumor_sample: str = None,
    x_columns_name_dict: dict = None,
    columns_to_drop: list = None,
    verbose: bool = True,
    raise_exception_on_sample_not_found: bool = False,
    is_matched: bool = None,
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
    is_matched: bool, optional
        Set is_matched column in the dataframe to be equal to the input value, if it is not None
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
                for x in str(header).split("\n"):
                    m = re.match(r"##tumor_sample=(.+)", x)
                    if m is not None:
                        tumor_sample = m.groups()[0]
                        logger.debug(f"Tumor sample name is {tumor_sample}")
                        break
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
                                rec.samples[tumor_sample]["AF"][0]
                                if tumor_sample
                                and "AF" in rec.samples[tumor_sample]
                                and len(rec.samples[tumor_sample]) > 0
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
                        + [c in rec.info for c in genomic_region_annotations]
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

    if is_matched is not None and isinstance(is_matched, bool):
        df_sig = df_sig.assign(is_matched=is_matched)

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
    substitution_error_rate=None,
    output_parquet=None,
    return_dataframes=False,
):
    """
    Read featuremap dataframes from several intersections of one featuremaps with several signatures, each is annotated
    with the signature name. Assumed to be the output of featuremap.intersect_featuremap_with_signature


    Parameters
    ----------
    intersected_featuremaps_parquet: list[str] or str
        list of featuremaps intesected with various signatures
    substitution_error_rate: str
        substitution error rate result generated from mrd.substitution_error_rate
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
            pd.read_parquet(f)
            .assign(
                cfdna=_get_sample_name_from_file_name(f, split_position=0),
                signature=_get_sample_name_from_file_name(f, split_position=1),
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

    if substitution_error_rate is not None:
        logger.debug("Merging with SNP error rate")
        df_sub_error_rate = (
            pd.read_hdf(substitution_error_rate, "/motif_1").set_index(["ref_motif", "alt"]).filter(regex="error_rate_")
        )
        df_int = df_int.merge(
            df_sub_error_rate,
            left_on=["ref_motif", "alt"],
            right_index=True,
        )

    logger.debug("Setting ref/alt direction to match reference and not read")
    if "X_FLAGS" not in df_int.columns:
        raise ValueError("X_FLAGS not found in dataframe - cannot determine read strand")
    is_reverse = (df_int["X_FLAGS"] & 16).astype(bool)
    for column in ("ref", "alt", "ref_motif", "alt_motif"):
        df_int.loc[:, column] = df_int[column].where(is_reverse, df_int[column].apply(revcomp))

    left_motif_reverse = df_int["left_motif"].apply(revcomp)
    right_motif_reverse = df_int["right_motif"].apply(revcomp)
    df_int.loc[:, "left_motif"] = df_int["left_motif"].where(is_reverse, right_motif_reverse)
    df_int.loc[:, "right_motif"] = df_int["right_motif"].where(is_reverse, left_motif_reverse)
    df_int = df_int.sort_index()
    df_int.columns = [x.replace("-", "_") for x in df_int.columns]
    if output_parquet is not None:
        df_int.reset_index().to_parquet(output_parquet)
    if return_dataframes:
        return df_int
    return None


def intersect_featuremap_with_signature(
    featuremap_file,
    signature_file,
    output_intersection_file=None,
    is_matched=None,
    add_info_to_header=True,
    overwrite=True,
    complement=False,
):
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
    is_matched: bool, optional
        If defined and tag the file name accordingly as "*.matched.vcf.gz" or "*.control.vcf.gz"
    add_info_to_header: bool, optional
        Add line to header to indicate this function ran (default True)
    overwrite: bool, optional
        Force rewrite of output (if false and output file exists an OSError will be raised). Default True.
    complement: bool, optional
        If True, only retain features that do not intersect with signature file - meant for removing germline variants
        from featuremap (default False)


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
        if is_matched is None:
            type_name = ""
        elif is_matched:
            type_name = ".matched"
        else:
            type_name = ".control"
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
    # build a set of all signature entries, including alts and ref
    signature_entries = set()
    with pysam.VariantFile(signature_file) as f_sig:
        for rec in f_sig:
            signature_entries.add((rec.chrom, rec.pos, rec.ref, rec.alts))
    # Only write entries from featuremap to intersection file if they appear in the signature with the same ref&alts
    try:
        with pysam.VariantFile(featuremap_file) as f_feat:
            header = f_feat.header
            if add_info_to_header is not None:
                header.add_line(
                    "##File:Description=This file is an intersection of a featuremap with a somatic mutation signature"
                )
                header.add_line(f"##python_cmd:intersect_featuremap_with_signature=python {' '.join(sys.argv)}")
                if is_matched is not None:
                    if is_matched:
                        header.add_line("##Intersection:type=matched (signature and featuremap from the same patient)")
                    else:
                        header.add_line(
                            "##Intersection:type=control (signature and featuremap not from the same patient)"
                        )
            with pysam.VariantFile(output_intersection_file_vcf + ".tmp", "w", header=header) as f_int:
                for rec in f_feat:
                    if ((not complement) and ((rec.chrom, rec.pos, rec.ref, rec.alts) in signature_entries)) or (
                        complement and ((rec.chrom, rec.pos, rec.ref, rec.alts) not in signature_entries)
                    ):
                        f_int.write(rec)
        os.rename(output_intersection_file_vcf + ".tmp", output_intersection_file_vcf)
    finally:
        if "output_intersection_file_vcf" in locals() and os.path.isfile(output_intersection_file_vcf + ".tmp"):
            os.remove(output_intersection_file_vcf + ".tmp")
    # index output
    pysam.tabix_index(output_intersection_file_vcf, preset="vcf", force=True)  # this also compressed the vcf to vcf.gz
    assert os.path.isfile(output_intersection_file_vcf + ".gz")


# pylint: disable=too-many-arguments
def featuremap_to_dataframe(
    featuremap_vcf: str,
    output_file: str = None,
    reference_fasta: str = None,
    motif_length: int = 4,
    report_bases_in_synthesis_direction: bool = True,
    info_fields_override: dict = None,
    format_fields: dict = None,
    show_progress_bar: bool = False,
    flow_order: str = DEFAULT_FLOW_ORDER,
    is_matched: bool = None,
    default_int_fillna_value: int = 0,
):
    """
    Converts featuremap in vcf format to dataframe
    if reference_fasta, annotates with motifs of length "motif_length"
    if flow_order is also given, annotates cycle skip status per entry


    Parameters
    ----------
    featuremap_vcf: str
        featuremap file generated by "gatk FeatureMap"
    output_file: str
        file path to save the featuremap dataframe in parquet format (in it does not end with .parquet it will be added)
    reference_fasta: str
        reference genome used to generate the bam that the featuremap was generated from, if not None (default) the
        entries in the featuremap are annorated for motifs with the length of the next parameters from either side
    motif_length: int
        default 4
    report_bases_in_synthesis_direction: bool
        featuremap entries are reported for the sense strand regardless of read diretion. If True (default), the ref and
        alt columns are reverse complemented for reverse strand reads (also motif if calculated).
    info_fields_override: dict
        Override info fields to extract from featuremap instead of the defaults:
        "X_CIGAR", "X_EDIST", "X_FC1", "X_FC2", "X_FILTERED_COUNT", "X_FLAGS", "X_LENGTH", "X_MAPQ", "X_READ_COUNT",
        "X_RN", "X_INDEX", "X_SCORE", "rq",
        The keys are the INFO field names and the values are the dtypes in the dataframe
    format_fields: dict
        Extra format fields to extract from vcf
        The keys are the FORMAT field names and the values are the dtypes in the dataframe
    default_int_fillna_value: int
        Used to fill missing values in integer columns, default 0
    show_progress_bar: bool
        displays tqdm progress bar of number of lines read (not in percent)
    flow_order: str
        flow order
    is_matched: bool, optional
        Set is_matched column in the dataframe to be equal to the input value, if it is not None

    Returns
    -------
        featuremap dataframe

    Raises
    ------
    ValueError
        If X_FLAGS is missing or by an internal function


    """
    # Define fields to include in the output DataFrame
    if info_fields_override is not None:
        info_fields = info_fields_override
    else:
        info_fields = default_featuremap_info_fields
    info_fields_list = list(info_fields.keys())

    if format_fields is None:
        format_fields = {}
        format_fields_list = []
    else:
        format_fields_list = list(format_fields.keys())

    # Read in the VCF file using pysam, extract entries, and store in a DataFrame
    with pysam.VariantFile(featuremap_vcf) as variant_file:
        # Create a generator that maps each variant to a defaultdict with the desired fields
        vfi = map(  # pylint: disable=bad-builtin
            lambda x: defaultdict(
                lambda: None,
                x.info.items()
                + [
                    ("chrom", x.chrom),
                    ("pos", x.pos),
                    ("ref", x.ref),
                    ("alt", x.alts[0]),
                    ("qual", x.qual),
                    ("filter", x.filter.keys()[0] if len(x.filter.keys()) > 0 else None),
                ]
                + [(xf, x.info.get(xf, None)) for xf in info_fields_list]
                + [(ff, x.samples[0][ff]) for ff in format_fields_list],
            ),
            variant_file,
        )

        # Define the columns for the output DataFrame
        columns = ["chrom", "pos", "ref", "alt", "qual", "filter"] + info_fields_list + format_fields_list

        # Create the DataFrame using the generator and the defined columns
        df = pd.DataFrame(
            (
                [x[y] for y in columns]
                for x in tqdm(
                    vfi,
                    disable=not show_progress_bar,
                    desc="Reading and converting vcf file",
                )
            ),
            columns=columns,
        )

    # If additional fields were provided, convert them to the appropriate data type
    fields = {}
    for dictionary in (info_fields, format_fields):
        if dictionary is not None:
            fields.update(dictionary)
    if len(fields) > 0:
        # Fill in missing values with 0 for int fields, otherwise casting
        if "int" in fields.values():
            df = df.fillna({k: default_int_fillna_value for k, v in fields.items() if v == int})
        for k, v in fields.items():
            try:
                df = df.astype({k: v})
            except ValueError:
                logger.warning(f"Could not convert column {k} to {v}")

    # If True (default), determine the read strand of each substitution based on the X_FLAGS (SAM flags) field
    if report_bases_in_synthesis_direction:
        # Ensure the X_FLAGS field is present in the dataframe
        if "X_FLAGS" not in df.columns:
            raise ValueError(
                "X_FLAGS not found in dataframe - cannot determine read strand, \
                did you mean to run with report_read_strand=False?"
            )
        # Determine whether the read is reverse strand or not based on the X_FLAGS field
        is_forward = ~(df["X_FLAGS"] & 16).astype(bool)
        # Reverse complement the ref and alt alleles if necessary to match the read direction
        for column in ("ref", "alt"):
            df[column] = df[column].where(is_forward, df[column].apply(revcomp))

    # If a reference fasta file was provided, extract flanking sequence motifs for each variant
    if reference_fasta is not None:
        # Annotate motifs around each substitution
        df = (
            get_motif_around(df.assign(indel=False), motif_length, reference_fasta)
            .drop(columns=["indel"])
            .astype({"left_motif": str, "right_motif": str})
        )

        # If specified, reverse the left and right motifs for variants on the reverse strand
        if report_bases_in_synthesis_direction:
            left_motif_reverse = df["left_motif"].apply(revcomp)
            right_motif_reverse = df["right_motif"].apply(revcomp)
            df["left_motif"] = df["left_motif"].where(is_forward, right_motif_reverse)
            df["right_motif"] = df["right_motif"].where(is_forward, left_motif_reverse)

        df["ref_motif"] = df["left_motif"].str.slice(-1) + df["ref"] + df["right_motif"].str.slice(0, 1)
        df["alt_motif"] = df["left_motif"].str.slice(-1) + df["alt"] + df["right_motif"].str.slice(0, 1)
        df = df.astype(
            {
                "chrom": CHROM_DTYPE,
                "ref": "category",
                "alt": "category",
                "ref_motif": "category",
                "alt_motif": "category",
                "left_motif": "category",
                "right_motif": "category",
            }
        )

        # If flow order was given annotate with cycle skip status
        if flow_order is not None:
            df_cskp = get_cycle_skip_dataframe(flow_order=flow_order)
            df = df.set_index(["ref_motif", "alt_motif"]).join(df_cskp).reset_index()

    # Calculate maximal softclip length
    if "X_CIGAR" in df:
        df.loc[:, "max_softclip_len"] = df["X_CIGAR"].apply(get_max_softclip_len)

    # Sort by position, then reset index becase saving without multi-index which breaks parquet
    df = df.set_index(["chrom", "pos"]).sort_index().reset_index()

    # Assign is_matched field if input was given
    if is_matched is not None and isinstance(is_matched, bool):
        df = df.assign(is_matched=is_matched)

    print(df)

    # Determine output file name
    if output_file is None:
        if featuremap_vcf.endswith(".vcf.gz"):
            output_file = featuremap_vcf[: -len(".vcf.gz")] + FileExtension.PARQUET.value
        else:
            output_file = f"featuremap_vcf{FileExtension.PARQUET.value}"
    elif not output_file.endswith(FileExtension.PARQUET.value):
        output_file = f"{output_file}{FileExtension.PARQUET.value}"

    # Save and return
    df.to_parquet(output_file)
    return df


def prepare_data_from_mrd_pipeline(
    intersected_featuremaps_parquet,
    matched_signatures_vcf_files=None,
    control_signatures_vcf_files=None,
    substitution_error_rate_hdf=None,
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
    substitution_error_rate_hdf: str
        snp error rate result generated from mrd.snp_error_rate, disabled (None) by default
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

    read_signature(
        matched_signatures_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        tumor_sample=tumor_sample,
        is_matched=True,
        return_dataframes=return_dataframes,
        concat_to_existing_output_parquet=False,
    )
    signature_dataframe = read_signature(
        control_signatures_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        tumor_sample=tumor_sample,
        is_matched=False,
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
