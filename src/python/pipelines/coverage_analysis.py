import pandas as pd
import numpy as np
import os
import sys
from os.path import join as pjoin, basename, dirname
from tempfile import TemporaryDirectory
import subprocess
from multiprocessing import cpu_count
from collections.abc import Iterable
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import argparse
import logging
import matplotlib.pyplot as plt

path = dirname(dirname(dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)
from python.auxiliary.cloud_sync import cloud_sync
from python.auxiliary.cloud_auth import get_gcs_token
from python.auxiliary.format import CHROM_DTYPE

# init logging
# create logger
logger = logging.getLogger("coverage_analysis")
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

# display defaults
SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 26
TITLE_SIZE = 36
FIGSIZE = (16, 8)
GRID = True
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=TITLE_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("axes", grid=GRID)  # is grid on
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=TITLE_SIZE)  # fontsize of the figure title
plt.rc("figure", figsize=FIGSIZE)  # size of the figure

# string constants
DEFAULT_INTERVALS = "s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_extended_intervals.tsv"
CHROM = "chrom"
CHROM_START = "chromStart"
CHROM_END = "chromEnd"
COVERAGE = "coverage"
PARQUET = "parquet"
HDF = "hdf"
H5 = "h5"
CSV = "csv"
TSV = "tsv"
ALL_BUT_X = "all_but_x"
ALL = "all"
CHROM_NUM = "chrom_num"
MERGED_REGIONS = "merged_regions"
BAM_EXT = ".bam"
CRAM_EXT = ".cram"
GCS_OAUTH_TOKEN = "GCS_OAUTH_TOKEN"


def calculate_and_bin_coverage(
    f_in: str,
    f_out: str = None,
    region: str = "chr9",
    merge_regions: bool = False,
    window: int = 100,
    min_bq: int = 0,
    min_mapq: int = 0,
    min_read_length: int = 0,
    max_read_length: int = None,
    ref_fasta: str = "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta",
    n_jobs: int = -1,
    progress_bar: bool = True,
    output_format=PARQUET,
    stop_on_errors=False,
):
    """
    Collect coverage in fixed windows across the genome or specific regions from an aligned bam/cram file
    The output is a dataframe with columns for chrom, chromStart, chromEnd and coverage (mean in each bin)

    Parameters
    ----------
    f_in: str
        input bam or cram file, or an Iterable of files
    f_out: str
        Path to which output dataframe will be written
        Interpreted as a base path if it contains no "." characters
        Can be None for output in the same directory as the input (or its cloud_sync)
        Can be an Iterable of file names if f_in is an Iterable
    region: str
        Genomic region in samtools format (i.e. chr9:1000000-2000000), can be None (default "chr9")
        Special allowed values:
            "all" for chr1,chr2,...,chr22,chrX
            "all_but_x" for chr1,chr2,...,chr22
    merge_regions: bool
        If True, merge output per region to a single dataframe
    window: int
        Number of base pairs to bin coverage by (default 100)
    min_bq: int
        Base quality theshold (default 0, samtools depth -q parameter)
    min_mapq: int
        Mapping quality theshold (default 0, samtools depth -Q parameter)
    min_read_length: int
        read length threshold (ignore reads shorter than <int>) (default 0, samtools depth -l parameter)
    max_read_length: int
        read length UPPER threshold (ignore reads longer than <int>)
    ref_fasta: str
        Reference fasta used for cram file compression, not used for bam inputs
        Default: "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"
    n_jobs: int
        Number of processes to run in parallel if f_in is an iterable
        Default -1 (joblib convention - the number of CPUs)
    progress_bar: bool
        Display progress bar for iterable f_in
    output_format: str
        File type of dataframe output, allowed values: PARQUET (default), "hdf", "h5", "csv", "tsv"
    stop_on_errors: bool
        If False (default) only warnings are raised
    -------

    Returns
        f_out: str or Iterable
            Output path(s) of saved dataframe(s) corresponding to specified input file(s)

    """
    # check inputs
    if not window > 0:
        raise ValueError(f"invalid windows size {window}")
    try:
        out = subprocess.check_output(["samtools", "--version"])
    except FileNotFoundError:
        raise ValueError("samtools executable not found in enrivonment")

    if output_format not in [PARQUET, HDF, H5, CSV, TSV]:
        raise ValueError(f"Unrecognized output_format {output_format}")
    region_name = "merged_regions"  # only used if merge_regions is True
    if region == ALL:
        region_name = ALL
        region = [f"chr{x}" for x in list(range(1, 23)) + ["X"]]
    elif region == ALL_BUT_X or region == "all_but_X":
        region_name = ALL_BUT_X
        region = [f"chr{x}" for x in range(1, 23)]
    logger.debug(f"Calculating coverage for file/s:\n{f_in}\n\nRegion/s:\n{region}")

    # the code below has a few options - either it got a single input file and a single region and max_read_length is
    # None and then the actual work is done and one output file is created, or it calls itself recursively according to
    # the following logic:
    # 1. If there are multiple input files, run recursively with a single call per input
    # 2. Otherwise, if a single input file but multiple regions are given, regions are called recursively
    # Additionally, if a single file and region were given but max_read_length is not None, then two calls with
    # different min_read_length thresholds are executed (samtools only has a minimal length option) and the outputs are
    # substracted.
    # The logic is executed below
    is_multiple_inputs = not isinstance(f_in, str) and isinstance(f_in, Iterable)
    is_multiple_regions = not isinstance(region, str) and isinstance(region, Iterable)
    is_max_length_set = max_read_length is not None
    is_single_input_file_and_region = not is_multiple_inputs and not is_multiple_regions
    if is_single_input_file_and_region:
        assert f_in.endswith(BAM_EXT) or f_in.endswith(CRAM_EXT)
        f_out = _get_output_file_name(
            f_in=f_in,
            f_out=f_out,
            min_bq=min_bq,
            min_mapq=min_mapq,
            min_read_length=min_read_length,
            max_read_length=max_read_length,
            region=region,
            window=window,
            output_format=output_format,
        )

        f_tmp = f_out + ".tmp"
        os.makedirs(dirname(f_out), exist_ok=True)
        if os.path.isfile(f_out):
            return f_out

        if not is_max_length_set:  # this clause is the actual implementation
            try:
                is_cram = f_in.endswith(CRAM_EXT)
                ref_fasta = cloud_sync(ref_fasta)

                samtools_depth_cmd = " ".join(
                    [
                        "samtools",
                        "depth",
                        "-a",
                        f" --reference {ref_fasta}" if is_cram else "",
                        f"-r {region}" if region is not None else "",
                        f"-q {min_bq}",
                        f"-Q {min_mapq}",
                        f"-l {min_read_length}",
                        f_in,
                    ]
                )
                cmd = f"{samtools_depth_cmd} > {f_tmp}"
                try:
                    token = (
                        get_gcs_token() if f_in.startswith("gs://") else ""
                    )  # only generate token if f_in is on gs

                    with TemporaryDirectory(
                        prefix="/data/tmp/tmp" if os.path.isdir("/data/") else None
                    ) as tmpdir:
                        logger.debug(f"Running command: {cmd}")
                        out = subprocess.check_output(
                            cmd,
                            shell=True,
                            cwd=tmpdir,
                            env={"PATH": os.environ["PATH"], GCS_OAUTH_TOKEN: token,},
                        )
                        logger.debug(f"Finished Running command: {cmd}")
                except subprocess.CalledProcessError:
                    warnings.warn(
                        f"Error running the command:\n{cmd}\nLikely a GCS_OAUTH_TOKEN issue"
                    )
                    if "out" in locals():
                        sys.stderr.write(f"{out}")
                    raise
                try:
                    logger.debug(f"Converting coverage tsv to dataframe")
                    df = pd.read_csv(f_tmp, sep="\t", header=None)
                    df.columns = [CHROM, CHROM_START, COVERAGE]
                    df = df.astype({CHROM: "category"})
                    df[COVERAGE] = (
                        df[COVERAGE]
                        .rolling(window=window, center=False, min_periods=1)
                        .mean()
                    )
                    df = df.iloc[:-1:window, :].reset_index()
                    df[CHROM_END] = df[CHROM_START] + window - 1
                except pd.errors.EmptyDataError:
                    if stop_on_errors:
                        raise
                    warnings.warn(f"Pandas parsing error on file {f_tmp}")

            finally:
                if "f_tmp" in locals() and os.path.isfile(f_tmp):
                    os.remove(f_tmp)
        else:  # max length mode - run recursively with two lower length thresholds and substract
            if max_read_length <= min_read_length:
                raise ValueError(
                    f"max_read_length (got {max_read_length}) must be larger than min_read_length (got {min_read_length})"
                )
            with TemporaryDirectory(prefix=pjoin(dirname(f_out), "tmp_")) as tmpdir:
                f_short = calculate_and_bin_coverage(
                    f_in,
                    f_out=tmpdir,
                    region=region,
                    merge_regions=merge_regions,
                    window=window,
                    min_bq=min_bq,
                    min_mapq=min_mapq,
                    min_read_length=min_read_length,
                    max_read_length=None,
                    ref_fasta=ref_fasta,
                    n_jobs=1,
                    progress_bar=False,
                    output_format=output_format,
                )
                f_long = calculate_and_bin_coverage(
                    f_in,
                    f_out=tmpdir,
                    region=region,
                    merge_regions=merge_regions,
                    window=window,
                    min_bq=min_bq,
                    min_mapq=min_mapq,
                    min_read_length=max_read_length,
                    max_read_length=None,
                    ref_fasta=ref_fasta,
                    n_jobs=1,
                    progress_bar=False,
                    output_format=output_format,
                )
                df, _ = _read_dataframe(f_short)
                df_long, _ = _read_dataframe(f_long)
                df = df[[CHROM, CHROM_START, CHROM_END, COVERAGE]]
                df[COVERAGE] = df[COVERAGE] - df_long[COVERAGE]

        # save output
        if "df" in locals():
            df = df[[CHROM, CHROM_START, CHROM_END, COVERAGE]]
            _save_datframe(df, f_out, output_format)
        return f_out
    elif is_multiple_inputs:
        if any(
            [x.endswith(CRAM_EXT) for x in f_in]
        ):  # download reference here if needed
            ref_fasta = cloud_sync(ref_fasta)
        if isinstance(f_out, str):
            f_out = [f_out] * len(f_in)  # yield f_out to sub functions
        elif f_out is None:
            f_out = [None] * len(f_in)  # yield None to sub functions
        if not len(f_in) == len(f_out):
            raise ValueError(
                f"Number of input files and output paths must be equal, got len(f_in)={len(f_in)}, len(f_out)={len(f_out)}"
            )
        # set number of jobs - we don't want to exceed total jobs in recursive calls
        n_jobs_actual = n_jobs if n_jobs > 0 else cpu_count() + n_jobs
        n_sub_jobs = -1 if n_jobs == -1 else max(1, n_jobs_actual // len(f_in))
        return Parallel(n_jobs=n_jobs)(
            delayed(calculate_and_bin_coverage)(
                f,
                f_out=fo,
                region=region,
                merge_regions=merge_regions,
                window=window,
                min_bq=min_bq,
                min_mapq=min_mapq,
                min_read_length=min_read_length,
                max_read_length=max_read_length,
                ref_fasta=ref_fasta,
                n_jobs=n_sub_jobs,
                progress_bar=False,
                output_format=output_format,
            )
            for f, fo in tqdm(zip(f_in, f_out), disable=not progress_bar)
        )
    elif is_multiple_regions:
        if merge_regions:
            f_out_merged = _get_output_file_name(
                f_in=f_in,
                f_out=f_out,
                min_bq=min_bq,
                min_mapq=min_mapq,
                min_read_length=min_read_length,
                max_read_length=max_read_length,
                region=region_name,
                window=window,
                output_format=output_format,
            )
            if os.path.isfile(
                f_out_merged
            ):  # merged file already exists here so we do nothing
                return f_out_merged
        f_out_list = Parallel(n_jobs=n_jobs)(
            delayed(calculate_and_bin_coverage)(
                f_in,
                f_out=f_out,
                region=r,
                window=window,
                min_bq=min_bq,
                min_mapq=min_mapq,
                min_read_length=min_read_length,
                max_read_length=max_read_length,
                ref_fasta=ref_fasta,
                n_jobs=1,
                progress_bar=False,
                output_format=output_format,
            )
            for r in tqdm(region, disable=not progress_bar)
        )
        if merge_regions:
            f_out = f_out_merged
            logger.debug(f"Merging coverage dataframes")
            df_merged = pd.concat(
                (
                    _read_dataframe(f)[0]
                    for f in tqdm(
                        f_out_list, total=len(f_out_list), desc="Merging dataframes"
                    )
                )
            )
            logger.debug(f"Merging of coverage dataframes done")

            def _f(x):
                try:
                    x = int(x.replace("chr", ""))
                except ValueError:
                    x = 100  # > 22
                return x

            df_merged[CHROM_NUM] = df_merged[CHROM].apply(_f)
            df_merged = (
                df_merged.sort_values([CHROM_NUM, CHROM_START])
                .drop(columns=[CHROM_NUM])
                .reset_index(drop=True)
            )
            logger.debug(f"Saving coverage dataframe to {f_out}")
            _save_datframe(df_merged, f_out, output_format)
            for f in f_out_list:
                if os.path.isfile(f):
                    os.remove(f)
            return f_out
        else:
            return f_out_list
    else:
        raise ValueError("Internal error - the code is never supposed to reach here")


def _get_output_file_name(
    f_in,
    f_out,
    min_bq,
    min_mapq,
    min_read_length,
    max_read_length,
    region,
    window,
    output_format,
):
    if f_out is None or "." not in basename(f_out):
        local_f_in = cloud_sync(f_in, dry_run=True)
        if f_out is None:
            local_dirname = dirname(local_f_in)
        else:
            local_dirname = f_out
        extra_extensions = "".join(
            [
                f".q{min_bq}" if min_bq > 0 else "",
                f".Q{min_mapq}" if min_mapq > 0 else "",
                f".l{min_read_length}" if min_read_length > 0 else "",
                f".L{max_read_length}" if max_read_length is not None else "",
            ]
        )
        out_basename = pjoin(local_dirname, basename(f_in).split(".")[0])
        region_str = "" if region is None else "." + region.replace(":", "_")
        f_out = (
            f"{out_basename}{region_str}.w{window}{extra_extensions}.{output_format}"
        )
    return f_out


def _read_dataframe(input_dataframe_file):
    # read coverage_dataframe
    if input_dataframe_file.endswith(".parquet"):
        df = pd.read_parquet(input_dataframe_file)
        input_format = PARQUET
    elif input_dataframe_file.endswith(".h5") or input_dataframe_file.endswith(".hdf"):
        df = pd.read_hdf(input_dataframe_file)
        input_format = H5
    elif input_dataframe_file.endswith(".csv"):
        df = pd.read_csv(input_dataframe_file)
        input_format = CSV
    elif input_dataframe_file.endswith(".tsv"):
        df = pd.read_csv(input_dataframe_file, sep="\t")
        input_format = TSV
    else:
        raise ValueError(
            f"""Could not interpret extension of {input_dataframe_file}\n
Should be one of {PARQUET}, {HDF}, {H5}, {CSV}, {TSV} """
        )
    return df, input_format


def _save_datframe(df, f_out, output_format):
    if output_format == PARQUET:
        df.to_parquet(f_out)
    elif output_format == HDF or output_format == H5:
        df.to_hdf(f_out, "data")
    elif output_format == CSV or output_format == TSV:
        df.to_csv(f_out, sep="," if output_format == CSV else "\t")
    else:
        raise ValueError(
            f"""Could not interpret output_format {output_format}\n
Should be one of {PARQUET}, {HDF}, {H5}, {CSV}, {TSV} """
        )


def create_coverage_annotations(
    coverage_dataframe: str,
    output_annotations_file: str = None,
    coverage_intervals_dict: str or dict = DEFAULT_INTERVALS,
    n_jobs: int = -1,
    progress_bar: bool = True,
    output_format: str = None,
):
    """
    Creates annotation dataframe matching a coverage file generated by calculate_and_bin_coverage
    The annotation intervals are given as input interval files
    The output is a dataframe with the same size and order as the input, and a binary column for each given annotation
    that is True if a bin is in that annotation interval and False otherwise.

    Note - when a window intersect an annotation interval partially it is counted True regardless of the size of overlap

    Parameters
    ----------
    coverage_dataframe: str
        Path to pandas dataframe created by calculate_and_bin_coverage
    output_annotations_file: str
        Output file to which annotations dataframe will be written, if None (default) will be created in the same
        directory as coverage_dataframe
    coverage_intervals_dict: str or dict
        Collection of Picard format intervals to use. Can be one of two formats:
        1. Dictionary with annotaion names for keys (will be used for column names in the output dataframe) and interval
        files (local or cloud) as values
        2. tsv file with column 'category' (name of annotation), 'file' (see below), and optionally 'order' (int)
        if tsv, the file name is assumed to start with "./" followed by a path relative to the tsv file
        currently the existing instances are:
        default "s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_rapidQC_intervals.tsv"
        alternative "s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_and_whole_exome_intervals.tsv"
    n_jobs: int
        Number of processes to run in parallel when intersecting coverage with intervals
        Default -1 (joblib convention - the number of CPUs)
    progress_bar: bool
        Display progress bar for iterable f_in
    output_format: str
        File type of dataframe output, allowed values: None (default - same as input), "parquet", "hdf", "h5", "csv", "tsv"

    output_annotations_file: str or Iterable
            Output path of saved annotations dataframe corresponding
    -------

    """
    if output_format not in [None, PARQUET, HDF, H5, CSV, TSV]:
        raise ValueError(f"Unrecognized output_format {output_format}")
    is_multiple_inputs = not isinstance(coverage_dataframe, str) and isinstance(
        coverage_dataframe, Iterable
    )
    if is_multiple_inputs:  # loop over inputs
        if isinstance(output_annotations_file, str):
            output_annotations_file = [output_annotations_file] * len(
                coverage_dataframe
            )  # yield f_out to sub functions
        elif output_annotations_file is None:
            output_annotations_file = [None] * len(
                coverage_dataframe
            )  # yield None to sub functions
        return Parallel(n_jobs=n_jobs)(
            delayed(create_coverage_annotations)(
                coverage_dataframe=f,
                output_annotations_file=fo,
                coverage_intervals_dict=coverage_intervals_dict,
                n_jobs=n_jobs if n_jobs == -1 else 1,
                progress_bar=progress_bar,
                output_format=output_format,
            )
            for f, fo in tqdm(
                zip(coverage_dataframe, output_annotations_file),
                disable=not progress_bar,
                total=min(len(coverage_dataframe), len(output_annotations_file)),
                desc="Generating coverage annotations",
            )
        )
    else:
        logger.debug(f"reading coverage dataframe {coverage_dataframe}")
        df, input_format = _read_dataframe(coverage_dataframe)
        logger.debug(f"coverage dataframe shape {df.shape}")

        if output_format is None:
            output_format = input_format
        # set output file name
        if output_annotations_file is None:
            output_annotations_file = pjoin(
                dirname(coverage_dataframe),
                ".".join(["annotations"] + basename(coverage_dataframe).split(".")),
            )
        if os.path.isfile(output_annotations_file):
            logger.debug(f"{output_annotations_file} already exists")
        else:
            os.makedirs(dirname(output_annotations_file), exist_ok=True)

            # fetch intervals
            df_coverage_intervals = _create_coverage_intervals_dataframe(
                coverage_intervals_dict
            )

            # start work
            with TemporaryDirectory(
                prefix="/data/tmp/tmp" if os.path.isdir("/data/tmp/") else None
            ) as tmpfile:
                logger.debug(f"working in temporary directory {tmpfile}")
                # write regions bed file
                bed_file = pjoin(
                    tmpfile,
                    f"regions.{'.'.join(basename(coverage_dataframe).split('.')[:-1])}.bed",
                )
                logger.debug(f"saving data to {bed_file}")
                n_chunks = 100
                ixs = np.array_split(df.index, n_chunks)
                for ix, subset in tqdm(
                    enumerate(ixs),
                    desc="Saving regions bed file",
                    total=n_chunks,
                    disable=not progress_bar,
                ):
                    df.loc[subset][[CHROM, CHROM_START, CHROM_END]].to_csv(
                        bed_file,
                        sep="\t",
                        index=False,
                        mode="w" if ix == 0 else "a",
                        header=True if ix == 0 else None,
                    )
                logger.debug(f"running bed file intersections")
                # create bed file per annotation
                out_intersected_beds = Parallel(n_jobs=n_jobs)(
                    delayed(_intersect_intervals)(interval, bed_file)
                    for interval in tqdm(
                        df_coverage_intervals["file"].values,
                        desc="Generating annotation bed files",
                        disable=not progress_bar,
                    )
                )
                logger.debug(f"bed file intersections done")
                df_coverage_intervals["out_intersected_bed"] = out_intersected_beds
                # read annotation bed files
                logger.debug(f"reading annotation bed files")
                df_list = [
                    _read_intersected_bed(
                        row["out_intersected_bed"], row["category"]
                    ).reset_index(level=CHROM_END)
                    for _, row in df_coverage_intervals.iterrows()
                ]
                # merge
                logger.debug(f"merging annotation bed dataframes")
                df_annotations = df_list[0]
                for df_tmp in tqdm(
                    df_list[1:],
                    desc="Merging intervals to single dataframe",
                    disable=not progress_bar,
                ):
                    # if df_tmp.shape[0] == 0:
                    #     continue
                    df_annotations = df_annotations.join(
                        df_tmp.drop(columns=[CHROM_END]), how="outer"
                    )
                logger.debug(f"setting index and sorting columns")
                df_annotations = (
                    df_annotations[~df_annotations.index.duplicated()]
                    .reset_index()
                    .set_index([CHROM, CHROM_START, CHROM_END])
                )
                df_annotations = df_annotations.reindex(
                    df.set_index([CHROM, CHROM_START, CHROM_END]).index
                ).fillna(False)
                df_annotations = df_annotations[
                    sorted(
                        df_annotations.columns,
                        key=lambda x: df_coverage_intervals.query(f"category=='{x}'")[
                            "order"
                        ].values[0],
                    )
                ]
            df_annotations = df_annotations.reset_index()
            # save
            logger.debug(f"saving annotations dataframe to {output_annotations_file}")
            _save_datframe(df_annotations, output_annotations_file, output_format)
        return output_annotations_file


def _create_coverage_intervals_dataframe(coverage_intervals_dict,):
    if isinstance(coverage_intervals_dict, str):
        if coverage_intervals_dict.endswith(TSV):
            sep = "\t"
        elif coverage_intervals_dict.endswith(CSV):
            sep = ","
        else:
            raise ValueError(
                f"""Unknown extension for input intervals dict file {coverage_intervals_dict}
Expected {TSV}/{CSV}"""
            )
        coverage_intervals_dict = cloud_sync(coverage_intervals_dict)
        df_coverage_intervals = pd.read_csv(coverage_intervals_dict, sep=sep)
        df_coverage_intervals["file"] = df_coverage_intervals.apply(
            lambda x: cloud_sync(
                pjoin(dirname(coverage_intervals_dict), x["file"][2:])
            ),
            axis=1,
        )
    elif isinstance(coverage_intervals_dict, dict):
        df_coverage_intervals = pd.DataFrame.from_dict(
            coverage_intervals_dict, orient="index"
        ).reset_index()
        df_coverage_intervals.columns = ["category", "file"]
        df_coverage_intervals["file"] = df_coverage_intervals.apply(
            lambda x: cloud_sync(x["file"]), axis=1,
        )
    else:
        raise ValueError(f"Invalid input {coverage_intervals_dict}")

    if "order" not in df_coverage_intervals:
        df_coverage_intervals = df_coverage_intervals.assign(
            order=range(df_coverage_intervals.shape[0])
        )
    return df_coverage_intervals


def _intersect_intervals(interval_file, regions_file, outdir=None):
    if outdir is None:
        outdir = dirname(regions_file)
    out_interval_bed = pjoin(
        outdir, ".".join(basename(interval_file).split(".")[:-1] + ["bed"])
    )
    out_intersected_bed = pjoin(
        outdir,
        ".".join(["regions"] + basename(interval_file).split(".")[:-1] + ["bed"]),
    )
    cmd_create_bed = (
        "echo "
        + f"\"chrom\\tchromStart\\tchromEnd\" > {out_interval_bed} && grep -v '^@' {interval_file} | sed 's/|/ /' "
        + ' | awk \'{print $1 "\\t" $2 "\\t" $3}\' '
        + f" >> {out_interval_bed}"
    )
    cmd_intersect = f"bedtools intersect -wa -a {regions_file} -b {out_interval_bed} > {out_intersected_bed}"
    if not os.path.isfile(out_interval_bed):
        subprocess.call(cmd_create_bed, shell=True)
    logger.debug(f"Running intersect command: {cmd_intersect}")
    if not os.path.isfile(out_intersected_bed):
        subprocess.Popen(
            cmd_intersect,
            env={
                "PATH": "/home/ubuntu/miniconda3/envs/genomics.py3/bin"
                + ":"
                + os.environ["PATH"]
            },
            shell=True,
        ).communicate()
    logger.debug(f"Finished executing intersect command: {cmd_intersect}")

    return out_intersected_bed


def _read_intersected_bed(intersected_bed, category_name):
    try:
        df = (
            pd.read_csv(intersected_bed, sep="\t", header=None,)
            .assign(**{category_name: True})
            .rename(columns={0: CHROM, 1: CHROM_START, 2: CHROM_END})
            .astype({CHROM: CHROM_DTYPE, CHROM_START: int, CHROM_END: int})
            .set_index([CHROM, CHROM_START, CHROM_END])
        )
    except pd.errors.EmptyDataError:
        df = (
            pd.DataFrame(columns=[CHROM, CHROM_START, CHROM_END])
            .set_index([CHROM, CHROM_START, CHROM_END])
            .assign(**{category_name: False})
        )
    return df


def _annotate_histogram_with_annotation_precentage(val_count):
    annotation_precentage = val_count.sum()
    annotation_precentage = annotation_precentage / annotation_precentage["Genome"]
    annotation_precentage.index.name = "annotation_precentage"
    annotation_precentage = annotation_precentage.T
    annotation_precentage[annotation_precentage > 1] = np.nan

    val_count = val_count.rename(
        columns={
            c: f"{c} ({annotation_precentage.loc[c]:.0%})"
            for c in annotation_precentage.index
        }
    )

    logger.debug(
        f"annotation_precentage generated:\n{annotation_precentage.to_string()}"
    )

    return val_count


def generate_histogram(
    df_coverage: str,
    df_annotations: str,
    out_path: str = None,
    max_coverage: int = 1000,
    normalize=True,
    annotate_columns_names=True,
    verbose=True
):

    if isinstance(df_coverage, str):
        df_coverage = _read_dataframe(df_coverage)[0]
    if isinstance(df_annotations, str):
        df_annotations = _read_dataframe(df_annotations)[0].set_index(
            [CHROM, CHROM_START, CHROM_END]
        )

    if verbose:
        logger.debug("generate_histogram started")

    val_count = pd.concat(
        (
            df_coverage[df_annotations[annotation].values]["coverage"]
            .round()
            .value_counts(normalize=normalize)
            .reindex(range(max_coverage + 1))
            .fillna(0)
            .rename(annotation)
            for annotation in df_annotations.columns
        ),
        axis=1,
    )
    val_count.index.name = "coverage"
    if annotate_columns_names:
        val_count = _annotate_histogram_with_annotation_precentage(val_count)
    if verbose:
        logger.debug(f"Histogram generated")

    if out_path is None:
        return val_count
    else:
        os.makedirs(out_path, exist_ok=True)
        if verbose:
            logger.debug(f"Saving data")
        coverage_stats_dataframes = pjoin(out_path, "coverage_stats.h5")
        if verbose:
            logger.debug(f"Saving histogram dataframe to {coverage_stats_dataframes}")
        val_count.to_hdf(coverage_stats_dataframes, key="histogram", mode="a")
        return coverage_stats_dataframes


def generate_stats_from_histogram(
    val_count, q=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.95]), out_path=None, verbose=True
):
    if isinstance(val_count, str) and os.path.isfile(val_count):
        val_count = pd.read_hdf(val_count, key="histogram")
    df_precentiles = pd.concat(
        (
            val_count.apply(lambda x: np.interp(q, np.cumsum(x), val_count.index)),
            val_count.apply(
                lambda x: np.average(val_count.index, weights=x)
                if x.sum() > 0
                else np.nan
            )
            .to_frame()
            .T,
        ),
        sort=False,
    )
    df_precentiles.index = pd.Index(
        data=[f"Q{int(qq * 100)}" for qq in q] + ["mean"], name="statistic"
    )

    genome_median = df_precentiles.loc["Q50"].filter(regex="Genome").values[0]
    selected_percentiles = (
        df_precentiles.loc[[f"Q{q}" for q in [5, 10, 50]]]
        .rename(index={"Q50": "median coverage"})
        .rename(index={f"Q{q}": f"{q}th percentile" for q in [5, 10, 50]})
    )
    selected_percentiles.loc[
        "median coverage (normalized to median genome coverage)"
    ] = (selected_percentiles.loc["median coverage"] / genome_median)
    df_stats = pd.concat(
        (
            selected_percentiles,
            pd.concat(
                (
                    (val_count.loc[(genome_median * 0.5).round().astype(int) :] * 100)
                    .sum()
                    .rename("% > 0.5 median of the genome")
                    .to_frame()
                    .T,
                    (val_count.loc[(genome_median * 0.25).round().astype(int) :] * 100)
                    .sum()
                    .rename("% > 0.25 median of the genome")
                    .to_frame()
                    .T,
                    (val_count.loc[10:] * 100)
                    .sum()
                    .rename("% bases with coverage >= 10x")
                    .to_frame()
                    .T,
                    (val_count.loc[20:] * 100)
                    .sum()
                    .rename("% bases with coverage >= 20x")
                    .to_frame()
                    .T,
                )
            ),
        )
    )
    if verbose:
        logger.debug(f"Generated stats:\n{df_stats.iloc[:, :10].to_string()}")

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        logger.debug(f"Saving data")
        if "." in basename(out_path):
            coverage_stats_dataframes = out_path
        else:
            coverage_stats_dataframes = pjoin(out_path, "coverage_stats.h5")
        if verbose:
            logger.debug(f"Saving dataframes to {coverage_stats_dataframes}")
        df_stats.to_hdf(coverage_stats_dataframes, key="stats", mode="a")
        df_precentiles.to_hdf(coverage_stats_dataframes, key="percentiles", mode="a")
        return coverage_stats_dataframes

    return df_precentiles, df_stats


def generate_coverage_boxplot(df_percentiles, color_group=None, out_path=None, title=""):
    if isinstance(df_percentiles, str) and os.path.isfile(df_percentiles):
        df_percentiles = pd.read_hdf(df_percentiles, key="percentiles")
    df_percentiles_norm = (
        df_percentiles / df_percentiles.loc["Q50"].filter(regex="Genome").values[0]
    )

    if color_group is None:
        color_group = range(df_percentiles.shape[1])
    elif isinstance(color_group, str) and color_group.endswith(TSV):
        # In this clause we either download the TSV or assume it's a dictionary with the right annotation values as keys
        color_group = pd.read_csv(cloud_sync(color_group), sep="\t",)["color_group"]

    color_list = [
        "#3274a1",
        "#e1812c",
        "#3a923a",
        "#c03d3e",
        "#9372b2",
        "blue",
        "orange",
        "green",
        "red",
        "purple",
    ]

    bxp = [
        {**v, **{"label": k}}
        for k, v in df_percentiles_norm.rename(
            {"Q50": "med", "Q25": "q1", "Q75": "q3", "Q5": "whislo", "Q95": "whishi"}
        )
        .to_dict()
        .items()
    ]

    logger.debug(f"Generating boxplot")
    plt.figure(figsize=(20, 8))
    fig = plt.gcf()
    ax = plt.gca()

    patches = ax.bxp(
        bxp, widths=0.7, showfliers=False, showmeans=True, patch_artist=True
    )
    ax.set_title(title)

    for j, bx in enumerate(bxp):
        plt.text(
            j + 1,
            bx["med"] + 0.03,
            f"{bx['med']:.2f}",
            horizontalalignment="center",
            fontsize=12,
        )
        plt.text(
            j + 1,
            bx["whislo"] - 0.06,
            f"{bx['whislo']:.2f}",
            horizontalalignment="center",
            fontsize=12,
        )

    _ = plt.xticks(rotation=90)
    xticks = ax.get_xticklabels()
    plt.ylim(-0.1, 2)
    plt.grid(axis="x")
    label = plt.ylabel("Coverage relative to median")
    text = plt.text(
        1,
        1.65,
        """Boxplot shows coverage percentiles (25th/50th/75/th - box, 5th/95th - whiskers)
Mean shown in triangle marker and numbers shown for median and 5th percentile
Calculated on chr9 unless noted otherwise (WG - Whole Genome)""",
        fontsize=12,
        bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=1"),
    )

    for j, b in enumerate(patches["boxes"]):
        b.set(color=color_list[color_group[j]])
        b.set_edgecolor("k")
        b.set_linewidth(2)

    for c in patches["caps"] + patches["medians"]:
        c.set_linewidth(2)
        c.set_color("k")

    for c in patches["means"]:
        c.set_linewidth(2)
        c.set_markerfacecolor("k")
        c.set_markeredgecolor("k")
        c.set_alpha(0.3)
    plt.tight_layout()

    if out_path is not None:
        if "." in basename(out_path):
            coverage_plot = out_path
        else:
            coverage_plot = pjoin(out_path, "coverage_boxplot.png")
        logger.debug(f"Saving coverage boxplot to {coverage_plot}")
        fig.savefig(
            coverage_plot,
            dpi=300,
            bbox_extra_artists=xticks + [label, text],
            bbox_inches="tight",
        )
        plt.close()
        return coverage_plot

    return fig


def generate_stats_and_plots(
    df_coverage: str,
    df_annotations: str,
    out_path: str = None,
    max_coverage: int = 1000,
    color_group: str or dict = DEFAULT_INTERVALS,
):
    """Generates coverage statistics and boxplot for a given coverage and annotations dataframes generated by 
    calculate_and_bin_coverage and create_coverage_annotations respectively

    Parameters
    ----------
    df_coverage: str
        Path to pandas dataframe created by calculate_and_bin_coverage, or the dataframe itself
    df_annotations: str
        Path to pandas dataframe created by create_coverage_annotations (with df_coverage as input), or the dataframe
    out_path: str
        Path to which results will be saved. If None (default), results are not saved but returned
    max_coverage: int
        Maximal coverage in the histogram (default 1000)
    color_group:
        Either an intervals tsv file containing a color_group column (like the default option -
        s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_rapidQC_intervals.tsv )
        Or a dictionary with annotation names (from df_annotations) as keys and color group (integer value) as values
        If None, each annotation gets a different color group


    Returns
    -------
        if out_path is None and no files were saved, returns fig (boxplot), df_stats, df_precentiles, val_count (hist)

        otherwise return coverage_stats_dataframes (stats hdf file with keys stats, percentiles, histogram),
        and coverage_plot (png file)

    """
    val_count = generate_histogram(
        df_coverage, df_annotations, out_path=None, max_coverage=max_coverage,
    )

    coverage_stats_dataframes = generate_stats_from_histogram(
        val_count, out_path=out_path
    )
    if out_path is None:
        df_percentiles = coverage_stats_dataframes[0]
        df_stats = coverage_stats_dataframes[1]
    else:
        df_percentiles = pd.read_hdf(coverage_stats_dataframes, key="percentiles")

    fig = generate_coverage_boxplot(
        df_percentiles, color_group=color_group, out_path=out_path
    )

    if out_path is None:
        return df_stats, df_percentiles, val_count, fig
    else:
        return coverage_stats_dataframes, fig


def run_full_coverage_analysis(
    f_in: str,
    out_path: str,
    coverage_intervals_dict: str,
    region: str = "all_but_x",
    window: int = 1,
    min_bq: int = 0,
    min_mapq: int = 0,
    min_read_length: int = 0,
    max_read_length: int = None,
    ref_fasta: str = "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta",
    n_jobs: int = -1,
    progress_bar: bool = True,
    output_format=PARQUET,
    stop_on_errors=False,
):
    if (
        isinstance(f_in, Iterable)
        and isinstance(f_in[0], Iterable)
        and len(f_in) == 1
    ):
        f_in = f_in[0]

    if region == "chr9":
        if n_jobs < 0:
            n_jobs_ = cpu_count() + 1 + n_jobs
        else:
            n_jobs_ = n_jobs
        CHR9_LENGTH = 138_394_717
        n = np.linspace(0, CHR9_LENGTH + 1, n_jobs_).astype(int)
        region = [f"chr9:{n[j] + 1}-{n[j + 1]}" for j in range(len(n) - 1)]
    elif region == "all_but_x":
        pass
    else:
        raise ValueError(f"Invalid region {region}")
    coverage_dataframes = calculate_and_bin_coverage(
        f_in=f_in,
        f_out=out_path,
        region=region,
        merge_regions=False,
        window=window,
        min_bq=min_bq,
        min_mapq=min_mapq,
        min_read_length=min_read_length,
        max_read_length=max_read_length,
        ref_fasta=ref_fasta,
        n_jobs=n_jobs,
        progress_bar=progress_bar,
        output_format=output_format,
        stop_on_errors=stop_on_errors,
    )
    coverage_annotations = create_coverage_annotations(
        coverage_dataframes,
        coverage_intervals_dict=coverage_intervals_dict,
        n_jobs=n_jobs,
        progress_bar=progress_bar,
        output_format=output_format,
    )

    val_counts = [
        generate_histogram(
            df_coverage=coverage_dataframe,
            df_annotations=coverage_annotation,
            out_path=None,
            normalize=False,
            annotate_columns_names=False,
            verbose=False
        )
        for coverage_dataframe, coverage_annotation in tqdm(
            zip(coverage_dataframes, coverage_annotations),
            desc="Calculating stats",
            total=len(coverage_dataframes),
        )
    ]
    val_count = val_counts[0]
    for v in val_counts[1:]:
        val_count += v
    val_count = _annotate_histogram_with_annotation_precentage(val_count)
    val_count = (val_count / val_count.sum()).fillna(0)

    coverage_stats_dataframes = generate_stats_from_histogram(
        val_count, out_path=out_path
    )
    val_count.to_hdf(coverage_stats_dataframes, key="histogram", mode="a")
    df_percentiles = pd.read_hdf(coverage_stats_dataframes, key="percentiles")

    generate_coverage_boxplot(
        df_percentiles, color_group=coverage_intervals_dict, out_path=out_path, title=basename(f_in).split(".")[0]
    )


def call_calculate_and_bin_coverage(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    output = calculate_and_bin_coverage(
        f_in=args_in.input,
        f_out=args_in.output,
        region=args_in.region,
        merge_regions=args_in.m,
        window=args_in.window,
        min_bq=args_in.q,
        min_mapq=args_in.Q,
        min_read_length=args_in.l,
        max_read_length=args_in.L,
        ref_fasta=args_in.reference,
        n_jobs=args_in.jobs,
        progress_bar=not args_in.no_progress_bar,
        output_format=args_in.output_format,
        stop_on_errors=args_in.raise_errors,
    )
    if isinstance(output, str):
        sys.stdout.write(output)
    else:
        sys.stdout.write(os.linesep.join(output) + os.linesep)


def call_create_coverage_annotations(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    output = create_coverage_annotations(
        coverage_dataframe=args_in.input,
        output_annotations_file=args_in.output,
        coverage_intervals_dict=args_in.coverage_intervals,
        n_jobs=args_in.jobs,
        progress_bar=not args_in.no_progress_bar,
        output_format=args_in.output_format,
    )
    if isinstance(output, str):
        sys.stdout.write(output)
    else:
        sys.stdout.write(os.linesep.join(output) + os.linesep)


def call_generate_stats_and_plots(args_in):
    generate_stats_and_plots(
        df_coverage=args_in.input_coverage,
        df_annotations=args_in.input_annotations,
        out_path=args_in.output,
        max_coverage=args_in.max_coverage,
        color_group=args_in.color_group,
    )


def call_run_full_coverage_analysis(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    run_full_coverage_analysis(
        f_in=args_in.input,
        out_path=args_in.output,
        coverage_intervals_dict=args_in.coverage_intervals,
        region=args_in.region,
        window=args_in.window,
        min_bq=args_in.q,
        min_mapq=args_in.Q,
        min_read_length=args_in.l,
        max_read_length=args_in.L,
        ref_fasta=args_in.reference,
        n_jobs=args_in.jobs,
        progress_bar=not args_in.no_progress_bar,
        output_format=args_in.output_format,
        stop_on_errors=args_in.raise_errors,
    )
    sys.stdout.write("DONE" + os.linesep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_calculate = subparsers.add_parser(
        name="calculate",
        description="""Collect coverage in fixed windows across the genome or specific regions from an aligned bam/cram file
The output is a dataframe with columns for chrom, chromStart, chromEnd and coverage (mean in each bin)""",
    )
    parser_calculate.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="input bam or cram file (multiple files allowed, e.g. -i f1 f2 f3)",
    )
    parser_calculate.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output dataframe will be written
Interpreted as a base path if it contains no "." characters
Can be None for output in the same directory as the input (or its cloud_sync)
Can be an Iterable of file names if INPUT is an Iterable""",
    )
    parser_calculate.add_argument(
        "-r",
        "--region",
        type=str,
        default="chr9",
        help="""Genomic region in samtools format (i.e. chr9:1000000-2000000), can be None (default "chr9")""",
    )
    parser_calculate.add_argument(
        "--m",
        "--merge_regions",
        action="store_true",
        help="If True, merge output per region to a single dataframe",
    )
    parser_calculate.add_argument(
        "-w",
        "--window",
        type=int,
        default=100,
        help="Number of base pairs to bin coverage by (default 100)",
    )
    parser_calculate.add_argument(
        "-q",
        "-bq",
        type=int,
        default=0,
        help="Base quality theshold (default 0, samtools depth -q parameter)",
    )
    parser_calculate.add_argument(
        "-Q",
        "-mapq",
        type=int,
        default=0,
        help="Mapping quality theshold (default 0, samtools depth -Q parameter)",
    )
    parser_calculate.add_argument(
        "-l",
        type=int,
        default=0,
        help="read length threshold (ignore reads shorter than <int>) (default 0, samtools depth -l parameter)",
    )
    parser_calculate.add_argument(
        "-L",
        type=int,
        default=None,
        help="read length UPPER threshold (ignore reads longer than <int>)",
    )
    parser_calculate.add_argument(
        "--reference",
        type=str,
        default="gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta",
        help="Reference fasta used for cram file compression, not used for bam inputs",
    )
    parser_calculate.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="Number of processes to run in parallel if INPUT is an iterable (joblib convention - the number of CPUs)",
    )
    parser_calculate.add_argument(
        "--no_progress_bar",
        default=False,
        action="store_true",
        help="Do not display progress bar for iterable INPUT",
    )
    parser_calculate.add_argument(
        "-f",
        "--output_format",
        type=str,
        default=PARQUET,
        help=f"""File type of dataframe output, allowed values: {PARQUET} (default), {HDF}, {H5}, {CSV}, {TSV} """,
    )
    parser_calculate.add_argument(
        "--raise_errors",
        default=False,
        action="store_true",
        help="If False (default) only warnings are raised",
    )
    parser_calculate.set_defaults(func=call_calculate_and_bin_coverage)

    parser_annotate = subparsers.add_parser(
        name="annotate",
        description="""Creates annotation dataframe matching a coverage file generated by calculate_and_bin_coverage
/ "python coverage_analysis.py calculate". 
        
The annotation intervals are given as input interval files
The output is a dataframe with the same size and order as the input, and a binary column for each given annotation
that is True if a bin is in that annotation interval and False otherwise.

Note - when a window intersect an annotation interval partially it is counted True regardless of the size of overlap""",
    )
    parser_annotate.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="""Path to pandas dataframe created by calculate_and_bin_coverage / "python coverage_analysis.py calculate" """,
    )
    parser_annotate.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Output file to which annotations dataframe will be written, if None (default) will be created in the same directory as coverage_dataframe""",
    )
    parser_annotate.add_argument(
        "-c",
        "--coverage_intervals",
        type=str,
        default=DEFAULT_INTERVALS,
        help="""tsv file pointing to a dataframe detailing the various intervals""",
    )
    parser_annotate.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="Number of processes to run in parallel when intersecting coverage with intervals",
    )
    parser_annotate.add_argument(
        "--no_progress_bar",
        default=False,
        action="store_true",
        help="Do not display progress bar for iterable INPUT",
    )
    parser_annotate.add_argument(
        "-f",
        "--output_format",
        type=str,
        default=PARQUET,
        help=f"""File type of dataframe output, allowed values: {PARQUET} (default), {HDF}, {H5}, {CSV}, {TSV} """,
    )
    parser_annotate.set_defaults(func=call_create_coverage_annotations)

    parser_annotate = subparsers.add_parser(
        name="stats",
        description="""Generates coverage statistics and boxplot for a given coverage and annotations dataframes generated by 
    calculate_and_bin_coverage/"python coverage_analysis.py calculate" and create_coverage_annotations/
    "python coverage_analysis.py annotate" respectively  """,
    )
    parser_annotate.add_argument(
        "-i",
        "--input_coverage",
        type=str,
        help="""Path to pandas dataframe created by calculate_and_bin_coverage / "python coverage_analysis.py calculate" """,
    )
    parser_annotate.add_argument(
        "-a",
        "--input_annotations",
        type=str,
        help="""Path to pandas dataframe created by create_coverage_annotations / "python coverage_analysis.py annotate" """,
    )
    parser_annotate.add_argument(
        "-o",
        "--output",
        type=str,
        help="""Output path to which output plot and stats dataframes will be written""",
    )
    parser_annotate.add_argument(
        "-m",
        "--max_coverage",
        type=int,
        default=1000,
        help="""Maximal coverage in the histogram""",
    )
    parser_annotate.add_argument(
        "-c",
        "--color_group",
        type=str,
        default=DEFAULT_INTERVALS,
        help="""tsv file pointing to a dataframe detailing the various intervals""",
    )
    parser_annotate.set_defaults(func=call_generate_stats_and_plots)

    parser_full_analysis = subparsers.add_parser(
        name="full_analysis",
        description="""Run full coverage analysis of an aligned bam/cram file""",
    )
    parser_full_analysis.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="input bam or cram file (multiple files allowed, e.g. -i f1 f2 f3)",
    )
    parser_full_analysis.add_argument(
        "-o",
        "--output",
        type=str,
        help="""Path to which output dataframe will be written
    Interpreted as a base path if it contains no "." characters
    Can be None for output in the same directory as the input (or its cloud_sync)
    Can be an Iterable of file names if INPUT is an Iterable""",
    )
    parser_full_analysis.add_argument(
        "-c",
        "--coverage_intervals",
        type=str,
        default=DEFAULT_INTERVALS,
        help="""tsv file pointing to a dataframe detailing the various intervals""",
    )
    parser_full_analysis.add_argument(
        "-r",
        "--region",
        type=str,
        default="all_but_x",
        help="""Genomic region in samtools format - the only allowed values are "all_but_x" and "chr9" """,
    )
    parser_full_analysis.add_argument(
        "-w",
        "--window",
        type=int,
        default=1,
        help="Number of base pairs to bin coverage by (default 100)",
    )
    parser_full_analysis.add_argument(
        "-q",
        "-bq",
        type=int,
        default=0,
        help="Base quality theshold (default 0, samtools depth -q parameter)",
    )
    parser_full_analysis.add_argument(
        "-Q",
        "-mapq",
        type=int,
        default=0,
        help="Mapping quality theshold (default 0, samtools depth -Q parameter)",
    )
    parser_full_analysis.add_argument(
        "-l",
        type=int,
        default=0,
        help="read length threshold (ignore reads shorter than <int>) (default 0, samtools depth -l parameter)",
    )
    parser_full_analysis.add_argument(
        "-L",
        type=int,
        default=None,
        help="read length UPPER threshold (ignore reads longer than <int>)",
    )
    parser_full_analysis.add_argument(
        "--reference",
        type=str,
        default="gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta",
        help="Reference fasta used for cram file compression, not used for bam inputs",
    )
    parser_full_analysis.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="Number of processes to run in parallel if INPUT is an iterable (joblib convention - the number of CPUs)",
    )
    parser_full_analysis.add_argument(
        "--no_progress_bar",
        default=False,
        action="store_true",
        help="Do not display progress bar for iterable INPUT",
    )
    parser_full_analysis.add_argument(
        "-f",
        "--output_format",
        type=str,
        default=PARQUET,
        help=f"""File type of dataframe output, allowed values: {PARQUET} (default), {HDF}, {H5}, {CSV}, {TSV} """,
    )
    parser_full_analysis.add_argument(
        "--raise_errors",
        default=False,
        action="store_true",
        help="If False (default) only warnings are raised",
    )
    parser_full_analysis.set_defaults(func=call_run_full_coverage_analysis)

    args = parser.parse_args()
    args.func(args)
