from __future__ import annotations

import itertools
import os
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
from ugvc.dna.utils import revcomp
from ugvc.utils.consts import FileExtension
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe, get_motif_around


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


def read_signature(
    signature_vcf_files,
    output_parquet=None,
    coverage_bw_files=None,
    sample_name="tumor",
    x_columns_name_dict=None,
    columns_to_drop=None,
    verbose=True,
    raise_exception_on_sample_not_found=False,
):
    """
    Read signature (variant calling output, generally mutect) results to dataframe.

    signature_vcf_files: str or list[str]
        File name or a list of file names
    output_parquet: str, optional
        File name to save result to, default None
    coverage_bw_files: list[str], optional
        Coverage bigwig files generated with "coverage_analysis full_analysis", defualt None
    sample_name: str, optional
        sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to be the
        full sample name, but does have to return a unique result. Default: "tumor"
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

    Raises
    ------
    ValueError
        may be raised

    """
    if not isinstance(signature_vcf_files, str) and isinstance(signature_vcf_files, Iterable):
        logger.debug(f"Reading and merging signature files:\n {signature_vcf_files}")
        df_sig = pd.concat(
            (
                read_signature(
                    file_name,
                    output_parquet=None,
                    sample_name=sample_name,
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
        x_columns_name_dict = x_columns_name_dict or {
            "X-CSS": "cycle_skip_status",
            "X-GCC": "gc_content",
            "X-LM": "left_motif",
            "X-RM": "right_motif",
            "X_CSS": "cycle_skip_status",
            "X_GCC": "gc_content",
            "X_LM": "left_motif",
            "X_RM": "right_motif",
        }
        columns_to_drop = columns_to_drop or ["X-IC", "X_IC"]

        if verbose:
            logger.debug(f"x_columns_name_dict:\n{x_columns_name_dict}")
            logger.debug(f"columns_to_drop:\n{columns_to_drop}")

        entries = []
        logger.debug(f"Reading vcf file {signature_vcf_files}")

        with pysam.VariantFile(signature_vcf_files) as variant_file:
            tumor_sample = None
            x_columns = []

            for j, rec in enumerate(variant_file):
                if j == 0:
                    x_columns = [
                        k
                        for k in rec.info.keys()
                        if (k.startswith("X-") or k.startswith("X_"))
                        and k not in columns_to_drop
                        and k in x_columns_name_dict
                    ]
                    x_columns_name_dict = {k: x_columns_name_dict[k] for k in x_columns}

                    if verbose:
                        logger.debug(f"Reading x_columns: {x_columns}")
                if tumor_sample is None:
                    candidate_sample_name = list(filter(lambda x: sample_name in x, rec.samples.keys()))
                    if len(candidate_sample_name) == 1:
                        tumor_sample = candidate_sample_name[0]
                    elif raise_exception_on_sample_not_found:
                        if len(candidate_sample_name) == 0:
                            raise ValueError(
                                f"No sample contained input string {sample_name}, sample names: {rec.samples.keys()}"
                            )
                        if len(candidate_sample_name) > 1:
                            raise ValueError(
                                f"{len(candidate_sample_name)} samples contained input string {sample_name}, "
                                f"expected just 1. Sample names: {rec.samples.keys()}"
                            )
                    else:  # leave AF blank
                        tumor_sample = "UNKNOWN"

                entries.append(
                    tuple(
                        [
                            rec.chrom,
                            rec.pos,
                            rec.ref,
                            rec.alts[0],
                            rec.id,
                            (
                                rec.samples[tumor_sample]["AF"][0]
                                if tumor_sample != "UNKNOWN"
                                and "AF" in rec.samples[tumor_sample]
                                and len(rec.samples[tumor_sample]) > 0
                                else np.nan
                            ),
                            (
                                rec.samples[tumor_sample]["DP"]
                                if tumor_sample != "UNKNOWN" and "DP" in rec.samples[tumor_sample]
                                else np.nan
                            ),
                            "MAP_UNIQUE" in rec.info,
                            "LCR" in rec.info,
                            "EXOME" in rec.info,
                            rec.info["LONG_HMER"] if "LONG_HMER" in rec.info else np.nan,
                            rec.info["TLOD"][0]
                            if "TLOD" in rec.info and isinstance(rec.info["TLOD"], Iterable)
                            else np.nan,
                            rec.info["SOR"] if "SOR" in rec.info else np.nan,
                        ]
                        + [rec.info[c][0] if isinstance(rec.info[c], tuple) else rec.info[c] for c in x_columns]
                    )
                )
        if verbose:
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
                    "af",
                    "depth_tumor_sample",
                    "map_unique",
                    "lcr",
                    "exome",
                    "hmer",
                    "tlod",
                    "sor",
                ]
                + [x_columns_name_dict[c] for c in x_columns],
            )
            .reset_index(drop=True)
            .astype({"chrom": str, "pos": int})
            .set_index(["chrom", "pos"])
        )
        if verbose:
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
    df_sig["hmer"] = (
        df_sig[["hmer"]]
        .assign(
            hmer_calc=df_sig.apply(
                lambda row: _get_hmer_length(row["ref"], row["left_motif"], row["right_motif"]),
                axis=1,
            )
        )
        .max(axis=1)
        .fillna(-1)
        .astype(int)
    )

    logger.debug("Annotating with mutation type (ref->alt)")
    ref_is_c_or_t = df_sig["ref"].isin(["C", "T"])
    df_sig.loc[:, "mutation_type"] = (
        np.where(ref_is_c_or_t, df_sig["ref"], df_sig["ref"].apply(revcomp))
        + "->"
        + np.where(ref_is_c_or_t, df_sig["alt"], df_sig["alt"].apply(revcomp))
    )

    df_sig.columns = [x.replace("-", "_") for x in df_sig.columns]

    if output_parquet is not None:
        if verbose:
            logger.debug(f"Saving output signature/s to {output_parquet}")
        df_sig.reset_index().to_parquet(output_parquet)
    return df_sig


def read_intersection_dataframes(
    intersected_featuremaps_parquet,
    substitution_error_rate=None,
    output_parquet=None,
):
    """
    Read featuremap dataframes from several intersections of one featuremaps with several signatures, each is annotated
    with the signature name. Assumed to be the output of featuremap.intersect_featuremap_with_signature


    Parameters
    ----------
    intersected_featuremaps_parquet: list[str]
        list of featuremaps intesected with various signatures
    substitution_error_rate: str
        substitution error rate result generated from mrd.substitution_error_rate
    output_parquet: str
        File name to save result to, default None

    Returns
    -------
    dataframe: pd.DataFrame
        concatenated intersection dataframes

    Raises
    ------
    ValueError
        may be raised

    """
    logger.debug(f"Reading {len(intersected_featuremaps_parquet)} intersection featuremaps")
    df_int = pd.concat(
        (
            pd.read_parquet(f)
            .assign(signature=_get_sample_name_from_file_name(f, split_position=1))
            .reset_index()
            .astype(
                {
                    "chrom": str,
                    "pos": int,
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
    return df_int


def intersect_featuremap_with_signature(
    featuremap_file,
    signature_file,
    output_intersection_file,
    append_python_call_to_header=True,
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
    output_intersection_file: str
        Output vcf file, .vcf.gz or .vcf extension
    append_python_call_to_header: bool
        Add line to header to indicate this function ran (default True)
    overwrite: bool
        Force rewrite of output (if false and output file exists an OSError will be raised). Default True.
    complement: bool
        If True, only retain features that do not intersect with signature file - meant for removing germline variants
        from featuremap (default False)


    Raises
    ------
    OSError
        in case the file already exists and function executed with no overwrite=True

    """
    output_intersection_file_vcf = (
        output_intersection_file[:-3] if output_intersection_file.endswith(".gz") else output_intersection_file
    )
    if (not overwrite) and (os.path.isfile(output_intersection_file) or os.path.isfile(output_intersection_file_vcf)):
        raise OSError(f"Output file {output_intersection_file} already exists and overwrite flag set to False")
    # build a set of all signature entries, including alts and ref
    signature_entries = set()
    with pysam.VariantFile(signature_file) as f_sig:
        for rec in f_sig:
            signature_entries.add((rec.chrom, rec.pos, rec.ref, rec.alts))
    # Only write entries from featuremap to intersection file if they appear in the signature with the same ref&alts
    try:
        with pysam.VariantFile(featuremap_file) as f_feat:
            header = f_feat.header
            if append_python_call_to_header is not None:
                header.add_line(f"##python_cmd:intersect_featuremap_with_signature=python {' '.join(sys.argv)}")
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


def featuremap_to_dataframe(
    featuremap_vcf: str,
    output_file: str = None,
    reference_fasta: str = None,
    motif_length: int = 4,
    report_read_strand: bool = True,
    x_fields: list = None,
    show_progress_bar: bool = False,
    flow_order: str = DEFAULT_FLOW_ORDER,
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
    report_read_strand: bool
        featuremap entries are reported for the sense strand regardless of read diretion. If True (default), the ref and
        alt columns are reverse complemented for reverse strand reads (also motif if calculated).
    x_fields: list
        fields to extract from featuremap, if default (None) those are extracted:
        "X_CIGAR", "X_EDIST", "X_FC1", "X_FC2", "X_FILTERED-COUNT", "X_FLAGS", "X_LENGTH", "X_MAPQ", "X_READ-COUNT",
        "X_RN", "X_INDEX", "X_SCORE", "rq",
    show_progress_bar: bool
        displays tqdm progress bar of number of lines read (not in percent)
    flow_order: str
        flow order

    Returns
    -------
        featuremap dataframe

    Raises
    ------
    ValueError
        If X_FLAGS is missing or by an internal function


    """
    if x_fields is None:
        x_fields = [
            "X_CIGAR",
            "X_EDIST",
            "X_FC1",
            "X_FC2",
            "X_READ_COUNT",
            "X_FILTERED_COUNT",
            "X_FLAGS",
            "X_LENGTH",
            "X_MAPQ",
            "X_INDEX",
            "X_RN",
            "X_SCORE",
            "rq",
        ]

    with pysam.VariantFile(featuremap_vcf) as variant_file:
        vfi = map(  # pylint: disable=bad-builtin
            lambda x: defaultdict(
                lambda: None,
                x.info.items()
                + [
                    ("CHROM", x.chrom),
                    ("POS", x.pos),
                    ("REF", x.ref),
                    ("ALT", x.alts[0]),
                ]
                + [(xf, x.info[xf]) for xf in x_fields],
            ),
            variant_file,
        )
        columns = ["chrom", "pos", "ref", "alt"] + x_fields
        df = pd.DataFrame(
            (
                [x[y.upper() if y != "rq" else y] for y in columns]
                for x in tqdm(
                    vfi,
                    disable=not show_progress_bar,
                    desc="Reading and converting vcf file",
                )
            ),
            columns=columns,
        )

    if report_read_strand:
        if "X_FLAGS" not in df.columns:
            raise ValueError("X_FLAGS not found in dataframe - cannot determine read strand")
        is_reverse = (df["X_FLAGS"] & 16).astype(bool)
        for column in ("ref", "alt"):  # reverse value to match the read direction
            df[column] = df[column].where(is_reverse, df[column].apply(revcomp))

    if reference_fasta is not None:
        df = (
            get_motif_around(df.assign(indel=False), motif_length, reference_fasta)
            .drop(columns=["indel"])
            .astype({"left_motif": str, "right_motif": str})
        )

        if report_read_strand:
            left_motif_reverse = df["left_motif"].apply(revcomp)
            right_motif_reverse = df["right_motif"].apply(revcomp)
            df["left_motif"] = df["left_motif"].where(is_reverse, right_motif_reverse)
            df["right_motif"] = df["right_motif"].where(is_reverse, left_motif_reverse)

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

        if flow_order is not None:
            df_cskp = get_cycle_skip_dataframe(flow_order=flow_order)
            df = df.set_index(["ref_motif", "alt_motif"]).join(df_cskp).reset_index()

    df = df.set_index(["chrom", "pos"]).sort_index().reset_index()  # saving without multi-index which breaks parquet
    if output_file is None:
        if featuremap_vcf.endswith(".vcf.gz"):
            output_file = featuremap_vcf[: -len(".vcf.gz")] + FileExtension.PARQUET.value
        else:
            output_file = f"featuremap_vcf{FileExtension.PARQUET.value}"
    elif not output_file.endswith(FileExtension.PARQUET.value):
        output_file = f"{output_file}{FileExtension.PARQUET.value}"
    df.to_parquet(output_file)
    return df


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

    intersected_featuremaps_parquet: list[str]
        list of featuremaps intesected with various signatures
    signature_vcf_files: list[str]
        File name or a list of file names
    snp_error_rate_hdf: str
        snp error rate result generated from mrd.snp_error_rate, disabled (None) by default
    coverage_bw_files: list[str]
        Coverage bigwig files generated with "coverage_analysis full_analysis", disabled (None) by default
    sample_name: str
        sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to be the
        full sample name, but does have to return a unique result. Default: "tumor"
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

    intersection_dataframe = read_intersection_dataframes(
        intersected_featuremaps_parquet,
        substitution_error_rate=snp_error_rate_hdf,
        output_parquet=intersection_dataframe_fname,
    )
    signature_dataframe = read_signature(
        signature_vcf_files,
        coverage_bw_files=coverage_bw_files,
        output_parquet=signatures_dataframe_fname,
        sample_name=sample_name,
    )
    return signature_dataframe, intersection_dataframe


def concat_dataframes(dataframes: list, outfile: str, n_jobs: int = 1):
    df = pd.concat(Parallel(n_jobs=n_jobs)(delayed(pd.read_parquet)(f) for f in dataframes))
    df = df.sort_index()
    if isinstance(df.index, pd.MultiIndex):  # DataFrames are saved as not multi-indexed so parquet doesn't break
        df = df.reset_index()
    df.to_parquet(outfile)
    return df
