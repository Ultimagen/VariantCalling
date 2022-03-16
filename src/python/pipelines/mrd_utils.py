import sys
from collections.abc import Iterable

import pandas as pd
import numpy as np
import pysam
import itertools
from os.path import dirname, basename, join as pjoin, isfile
import os
import argparse
import pyfaidx
import logging
from tqdm import tqdm
from tempfile import TemporaryDirectory
import subprocess
import pyBigWig as bw

if __name__ == "__main__":
    import pathmagic

logger = logging.getLogger("mrd_utils")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


from python.utils import revcomp, get_cycle_skip_dataframe


def create_control_signature(
    signature_file,
    reference_fasta,
    control_signature_file_output=None,
    append_python_call_to_header=True,
    delta=1,
    min_distance=5,
    force_overwrite=True,
    progress_bar=False,
):
    """
    Creates a control signature that matches each SNP in the input signature vcf file with an adjacent position with
    the same trinucleotide motif, maintaining the same ref and alt composition. Non-SNP entries are ignored.
    Adds an ORIG_SEQ info in the output vcf indicating the position of the original variant this controls for.

    Parameters
    ----------
    signature_file:
        Input vcf file
    reference_fasta:
        Reference fasta file
        an index (.fai) file is expected to be in the same path
    control_signature_file_output:
        Output path, if None (default) the input file name is used with a ".control.vcf.gz" suffix
    append_python_call_to_header
        Add line to header to indicate this function ran (default True)
    delta
        How many bp to skip when searching for motifs, default 1
    min_distance
        minimum distance in bp from the original position
    force_overwrite
        Force rewrite tbi index of output (if false and output file exists an error will be raised). Default True.
    progress_bar
        Show progress bar (default False)

    Returns
    -------

    """
    ref = pyfaidx.Fasta(reference_fasta)

    assert delta > 0, f"Input parameter delta must be positive, got {delta}"
    if control_signature_file_output is None:
        control_signature_file_output = f"{signature_file}.control.vcf.gz".replace(
            ".vcf.gz.control.", ".control."
        )
    if (not force_overwrite) and os.path.isfile(control_signature_file_output):
        raise OSError(
            f"Output file {control_signature_file_output} already exists and force_overwrite flag set to False"
        )

    with TemporaryDirectory(prefix=control_signature_file_output) as tmpdir:
        tmp_file = pjoin(tmpdir, control_signature_file_output)
        with pysam.VariantFile(signature_file) as f_in:
            header = f_in.header
            header.info.add(
                "ORIG_POS", 1, "Integer", "Original position of the variant"
            )
            if append_python_call_to_header is not None:
                header.add_line(
                    f"##python_cmd:create_control_signature=python {' '.join(sys.argv)}"
                )
            with pysam.VariantFile(tmp_file, "w", header=header) as f_out:
                for rec in tqdm(
                    f_in, disable=not progress_bar, desc=f"Processing {signature_file}"
                ):
                    is_snv = (
                        len(rec.ref) == 1
                        and len(rec.alts) == 1
                        and len(rec.alts[0]) == 1
                    )
                    if not is_snv:
                        continue

                    chrom = rec.chrom
                    pos = rec.pos
                    motif = ref[chrom][pos - 2 : pos + 1].seq
                    assert (
                        rec.ref == motif[1]
                    ), f"Inconsistency in reference found!\n{rec.chrom} {rec.pos} ref={rec.ref} motif={motif}"

                    new_pos = pos + min_distance  # starting position
                    delta = int(delta)
                    while True:
                        new_motif = ref[chrom][new_pos - 2 : new_pos + 1].seq
                        if (
                            len(new_motif) < 3 or new_motif == "NNN"
                        ):  # reached the end of the chromosome or a reference gap
                            if (
                                delta < 0
                            ):  # if we already looked in the other direction, stop and give up this entry (very rare)
                                break
                            # start looking in the other direction
                            delta = -1 * delta
                            new_pos = pos - min_distance
                            continue
                        if motif == new_motif:  # motif matches
                            rec.pos = new_pos
                            rec.info["ORIG_POS"] = pos
                            f_out.write(rec)
                            break
                        new_pos += delta
        # sort because output can become unsorted and then cannot be indexed
        subprocess.call(
            f"bcftools sort -Oz -o {control_signature_file_output} {tmp_file}".split()
        )
    # index output
    pysam.tabix_index(
        control_signature_file_output, preset="vcf", force=force_overwrite
    )


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
        snp error rate result generated from featuremap.calculate_snp_error_rate
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
        snp error rate result generated from featuremap.calculate_snp_error_rate, disabled (None) by default
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
        pjoin(output_dir, f"{output_basename}.features.parquet")
        if output_dir is not None
        else None
    )
    signatures_dataframe_fname = (
        pjoin(output_dir, f"{output_basename}.signatures.parquet")
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


def call_create_control_signature(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    create_control_signature(
        signature_file=args_in.input,
        control_signature_file_output=args_in.output,
        reference_fasta=args_in.reference,
        progress_bar=args_in.progress_bar,
    )
    sys.stdout.write("DONE" + os.linesep)


def call_prepare_data_from_mrd_pipeline(args_in):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_create_control_signature = subparsers.add_parser(
        name="create_control_signature",
        description="""Run full coverage analysis of an aligned bam/cram file""",
    )
    parser_create_control_signature.add_argument(
        "-i", "--input", type=str, required=True, help="input signature vcf file",
    )
    parser_create_control_signature.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output bed file will be written. 
If None (default) the input file name is used with a ".control.vcf.gz" suffix""",
    )
    parser_create_control_signature.add_argument(
        "-r", "--reference", type=str, required=True, help="Reference fasta (local)",
    )
    parser_create_control_signature.add_argument(
        "--progress-bar", action="store_true", help="""Show progress bar""",
    )
    parser_create_control_signature.set_defaults(func=call_create_control_signature)

    parser_prepare_data_from_mrd_pipeline = subparsers.add_parser(
        name="prepare_data",
        description="""Aggregate the outputs from the MRDFeatureMap pipeline and prepare dataframes for analysis""",
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "-f",
        "--intersected-featuremaps",
        nargs="+",
        type=str,
        required=True,
        help="Input signature and featuemaps vcf files",
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "-s",
        "--signature-vcf",
        nargs="+",
        type=str,
        required=True,
        help="Input signature vcf files",
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "-e",
        "--snp-error-rate",
        type=str,
        default=None,
        required=False,
        help="SNP error rate hdf produced by featuremap.calculate_snp_error_rate",
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "-c",
        "--coverage-bw",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Coverage bigwig files generated with 'coverage_analysis full_analysis'",
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "--sample-name",
        type=str,
        required=False,
        default="tumor",
        help=""" sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to 
be the full sample name, but does have to return a unique result. Default: "tumor" """,
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="""Path to which output files will be written.""",
    )
    parser_prepare_data_from_mrd_pipeline.add_argument(
        "-b",
        "--output-basename",
        type=str,
        default=None,
        help="""Basename of output files that will be created.""",
    )
    parser_prepare_data_from_mrd_pipeline.set_defaults(
        func=call_prepare_data_from_mrd_pipeline
    )

    args = parser.parse_args()
    args.func(args)
