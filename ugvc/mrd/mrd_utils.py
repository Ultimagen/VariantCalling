import itertools
from collections.abc import Iterable
from os.path import basename

import numpy as np
import pandas as pd
import pyBigWig as bw
import pysam
from tqdm import tqdm

from ugvc import logger
from ugvc.dna.utils import revcomp


def _get_sample_name_from_file_name(f, split_position=0):
    """
    Internal formatting of filename for mrd pipeline
    Parameters
    ----------
    f
        file name
    split_position
        which position relative to splittinmg by "." the sample name is

    Returns
    -------

    """
    return (
        basename(f)
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


def _get_hmer_length(ref, left_motif, right_motif):
    """
    Calculate the length of the hmer the ref is contained in (limited by input motif length)

    Parameters
    ----------
    ref
        reference base (single base)
    left_motif
        left motif
    right_motif
        right motif

    Returns
    -------

    """

    left_motif_len = len(left_motif)
    x = np.array(
        [
            (k == ref, sum(1 for _ in v))
            for k, v in itertools.groupby(left_motif + ref + right_motif)
        ]
    )
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

    Parameters
    ----------
    signature_vcf_files
        File name or a list of file names
    output_parquet
        File name to save result to, default None
    coverage_bw_files
        Coverage bigwig files generated with "coverage_analysis full_analysis", defualt None
    sample_name
        sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to be the
        full sample name, but does have to return a unique result. Default: "tumor"
    x_columns_name_dict
        Dictionary of INFO fields starting with "X-" (custom UG fields) to read as keys and respective columns names
        as values, if None (default) this value if used:
        {"X-CSS": "cycle_skip_status","X-GCC": "gc_content","X-LM": "left_motif","X-RM": "right_motif"}
    columns_to_drop
        List of columns to drop, as values, if None (default) this value if used: ["X-IC"]
    verbose
        show verbose debug messages
    raise_exception_on_sample_not_found
        if True (default False) and the sample name could not be found to determine AF, raise a ValueError


    Returns
    -------

    """
    if not isinstance(signature_vcf_files, str) and isinstance(
        signature_vcf_files, Iterable
    ):
        logger.debug(f"Reading and merging signature files:\n{signature_vcf_files}")
        df_sig = pd.concat(
            (
                read_signature(
                    f,
                    output_parquet=None,
                    sample_name=sample_name,
                    x_columns_name_dict=x_columns_name_dict,
                    columns_to_drop=columns_to_drop,
                    verbose=j == 0,  # only verbose in first iteration
                )
                .assign(signature=_get_sample_name_from_file_name(f, split_position=0))
                .reset_index()
                .set_index(["signature", "chrom", "pos"])
                for j, f in enumerate(np.unique(signature_vcf_files))
            )
        )

    else:
        if x_columns_name_dict is None:
            x_columns_name_dict = {
                "X-CSS": "cycle_skip_status",
                "X-GCC": "gc_content",
                "X-LM": "left_motif",
                "X-RM": "right_motif",
                "X_CSS": "cycle_skip_status",
                "X_GCC": "gc_content",
                "X_LM": "left_motif",
                "X_RM": "right_motif",
            }
        if columns_to_drop is None:
            columns_to_drop = ["X-IC", "X_IC"]
        if verbose:
            logger.debug(f"x_columns_name_dict:\n{x_columns_name_dict}")
            logger.debug(f"columns_to_drop:\n{columns_to_drop}")

        entries = list()
        logger.debug(f"Reading vcf file {signature_vcf_files}")
        with pysam.VariantFile(signature_vcf_files) as f:
            tumor_sample = None
            x_columns = list()
            for j, rec in enumerate(f):
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
                    candidate_sample_name = list(
                        filter(lambda x: sample_name in x, rec.samples.keys())
                    )
                    if len(candidate_sample_name) == 1:
                        tumor_sample = candidate_sample_name[0]
                    elif raise_exception_on_sample_not_found:
                        if len(candidate_sample_name) == 0:
                            raise ValueError(
                                f"No sample contained input string {sample_name}, sample names: {rec.samples.keys()}"
                            )
                        if len(candidate_sample_name) > 1:
                            raise ValueError(
                                f"{len(candidate_sample_name)} samples contained input string {sample_name}, expected just 1. Sample names: {rec.samples.keys()}"
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
                                if tumor_sample != "UNKNOWN"
                                and "DP" in rec.samples[tumor_sample]
                                else np.nan
                            ),
                            "MAP_UNIQUE" in rec.info,
                            "LCR" in rec.info,
                            "EXOME" in rec.info,
                            rec.info["LONG_HMER"]
                            if "LONG_HMER" in rec.info
                            else np.nan,
                            rec.info["TLOD"][0]
                            if "TLOD" in rec.info
                            and isinstance(rec.info["TLOD"], Iterable)
                            else np.nan,
                            rec.info["SOR"] if "SOR" in rec.info else np.nan,
                        ]
                        + [
                            rec.info[c][0]
                            if isinstance(rec.info[c], tuple)
                            else rec.info[c]
                            for c in x_columns
                        ]
                    )
                )
        if verbose:
            logger.debug(f"Done reading vcf file {signature_vcf_files}")
            logger.debug(f"Converting to dataframe")
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
            logger.debug(f"Done converting to dataframe")

    df_sig = df_sig.sort_index()

    if (
        coverage_bw_files is not None and len(coverage_bw_files) > 0
    ):  # collect coverage per locus
        try:
            logger.debug(f"Reading input from bigwig coverage data")
            f_bw = [bw.open(x) for x in coverage_bw_files]
            df_list = list()
            for chrom, df_tmp in tqdm(df_sig.groupby(level="chrom")):
                if df_tmp.shape[0] == 0:
                    continue
                found_correct_file = False
                for f_bw_chrom in f_bw:
                    if chrom in f_bw_chrom.chroms():
                        found_correct_file = True
                        break
                if not found_correct_file:
                    raise ValueError(
                        f"Could not find a bigwig file with {chrom} in:\n{', '.join(coverage_bw_files)}"
                    )
                chrom_start = df_tmp.index.get_level_values("pos") - 1
                chrom_end = df_tmp.index.get_level_values("pos")
                df_list.append(
                    df_tmp.assign(
                        coverage=np.concatenate(
                            [
                                f_bw_chrom.values(chrom, x, y, numpy=True)
                                for x, y in zip(chrom_start, chrom_end)
                            ]
                        )
                    )
                )
        finally:
            if "f_bw" in locals():
                for f in f_bw:
                    f.close()
        df_sig = pd.concat(df_list).astype({"coverage": int})

    logger.debug("Calculating reference hmer")
    df_sig["hmer"] = (
        df_sig[["hmer"]]
        .assign(
            hmer_calc=df_sig.apply(
                lambda row: _get_hmer_length(
                    row["ref"], row["left_motif"], row["right_motif"]
                ),
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
        df_sig.to_parquet(output_parquet)
    return df_sig


def read_intersection_dataframes(
    intersected_featuremaps_parquet, snp_error_rate=None, output_parquet=None,
):
    """
    Read featuremap dataframes from several intersections of one featuremaps with several signatures, each is annotated
    with the signature name. Assumed to be the output of featuremap.intersect_featuremap_with_signature

    Parameters
    ----------
    intersected_featuremaps_parquet
        list of featuremaps intesected with various signatures
    snp_error_rate
        snp error rate result generated from mrd.snp_error_rate
    output_parquet
        File name to save result to, default None

    Returns
    -------
    dataframe

    """
    logger.debug(
        f"Reading {len(intersected_featuremaps_parquet)} intersection featuremaps"
    )
    df_int = pd.concat(
        (
            pd.read_parquet(f)
            .assign(signature=_get_sample_name_from_file_name(f, split_position=1))
            .reset_index()
            .astype({"chrom": str, "pos": int,})
            .set_index(["signature", "chrom", "pos"])
            for f in intersected_featuremaps_parquet
        )
    )

    if snp_error_rate is not None:
        logger.debug("Merging with SNP error rate")
        df_snp_error_rate = (
            pd.read_hdf(snp_error_rate, "/motif_1")
            .set_index(["ref_motif", "alt"])
            .filter(regex="error_rate_")
        )
        df_int = df_int.merge(
            df_snp_error_rate, left_on=["ref_motif", "alt"], right_index=True,
        )

    logger.debug("Setting ref/alt direction to match reference and not read")
    is_reverse = (df_int["X_FLAGS"] & 16).astype(bool)
    for c in ["ref", "alt", "ref_motif", "alt_motif"]:
        df_int.loc[:, c] = df_int[c].where(is_reverse, df_int[c].apply(revcomp))

    left_motif_reverse = df_int["left_motif"].apply(revcomp)
    right_motif_reverse = df_int["right_motif"].apply(revcomp)
    df_int.loc[:, "left_motif"] = df_int["left_motif"].where(
        is_reverse, right_motif_reverse
    )
    df_int.loc[:, "right_motif"] = df_int["right_motif"].where(
        is_reverse, left_motif_reverse
    )
    df_int = df_int.sort_index()
    df_int.columns = [x.replace("-", "_") for x in df_int.columns]
    if output_parquet is not None:
        df_int.to_parquet(output_parquet)
    return df_int
