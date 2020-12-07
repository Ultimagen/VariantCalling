import pandas as pd
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
from python.auxiliary.cloud_sync import cloud_sync


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
    output_format="parquet",
    stop_on_errors=False,
):
    """

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
        File type of dataframe output, allowed values: "parquet" (default), "hdf", "h5", "csv", "tsv"
    stop_on_errors: bool
        If False (default) only warnings are raised
    -------

    Returns
        f_out: str or Iterable
            Output path(s) of saved dataframe(s) corresponding to specified input file(s)

    """
    # check inputs
    try:
        out = subprocess.call(["samtools", "--version"])
    except FileNotFoundError:
        raise ValueError("samtools executable not found in enrivonment")

    if output_format not in ["parquet", "hdf", "h5", "csv", "tsv"]:
        raise ValueError(f"Unrecognized output_format {output_format}")
    if region == "all":
        region = [f"chr{x}" for x in list(range(1, 23)) + ["X"]]
    elif region == "all_but_x" or region == "all_but_X":
        region = [f"chr{x}" for x in range(1, 23)]
    multiple_inputs = not isinstance(f_in, str) and isinstance(f_in, Iterable)
    if multiple_inputs:  # loop over inputs
        if any(
            [x.endswith(".cram") for x in f_in]
        ):  # download reference here if needed
            ref_fasta = cloud_sync(ref_fasta)
        if isinstance(f_out, str):
            f_out = [f_out] * len(f_in)  # yield f_out to sub functions
        elif f_out is None:
            f_out = [None] * len(f_in)  # yield None to sub functions
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
    else:
        if not isinstance(region, str) and isinstance(
            region, Iterable
        ):  # loop over regions
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
                df_merged = pd.concat((_read_dataframe(f)[0] for f in f_out_list))

                def _f(x):
                    try:
                        x = int(x.replace("chr", ""))
                    except ValueError:
                        x = 100  # > 22
                    return x

                df_merged["chrom_num"] = df_merged["chrom"].apply(_f)
                df_merged = (
                    df_merged.sort_values(["chrom_num", "chromStart"])
                    .drop(columns=["chrom_num"])
                    .reset_index(drop=True)
                )
                f_out = _get_output_file_name(
                    f_in=f_in,
                    f_out=f_out,
                    min_bq=min_bq,
                    min_mapq=min_mapq,
                    min_read_length=min_read_length,
                    max_read_length=max_read_length,
                    region="merged_regions",
                    window=window,
                    output_format=output_format,
                )
                _save_datframe(df_merged, f_out, output_format)
                for f in f_out_list:
                    if os.path.isfile(f):
                        os.remove(f)
                return f_out
            else:
                return f_out_list
        else:
            try:
                assert f_in.endswith(".bam") or f_in.endswith(".cram")
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
                elif max_read_length is not None:
                    if max_read_length <= min_read_length:
                        raise ValueError(
                            f"max_read_length (got {max_read_length}) must be larger than min_read_length (got {min_read_length})"
                        )
                    with TemporaryDirectory(
                        prefix=pjoin(dirname(f_out), "tmp_")
                    ) as tmpdir:
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
                        df = df[["chrom", "chromStart", "chromEnd", "name"]]
                        df["name"] = df["name"] - df_long["name"]
                else:  # this clause is the actual implementation
                    is_cram = f_in.endswith(".cram")
                    ref_fasta = cloud_sync(ref_fasta)
                    gcs_token_cmd = "gcloud auth application-default print-access-token"

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

                    if not os.path.isfile(f_tmp):
                        try:
                            for var in [
                                "GOOGLE_APPLICATION_CREDENTIALS",
                                "CLOUDSDK_PYTHON",
                            ]:
                                if var not in os.environ:
                                    raise ValueError(f"{var} not in environment")
                            if f_in.startswith("gs://"):
                                token = subprocess.check_output(
                                    gcs_token_cmd.split(),
                                    cwd="/tmp",
                                    env={
                                        "GOOGLE_APPLICATION_CREDENTIALS": os.environ[
                                            "GOOGLE_APPLICATION_CREDENTIALS"
                                        ],
                                        "CLOUDSDK_PYTHON": os.environ[
                                            "CLOUDSDK_PYTHON"
                                        ],
                                    },
                                ).decode()
                            else:
                                token = ""
                        except subprocess.CalledProcessError:
                            raise ValueError("Error getting google storage token")
                        try:
                            stdout, stderr = subprocess.Popen(
                                cmd,
                                shell=True,
                                cwd="/tmp",
                                env={
                                    "GCS_OAUTH_TOKEN": token,
                                    "PATH": os.environ["PATH"],
                                },
                            ).communicate()
                        except:
                            if "stdout" in locals():
                                sys.stderr.write(
                                    f"Error running the command:\n{cmd}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}\n"
                                )
                            raise
                    try:
                        df = pd.read_csv(f_tmp, sep="\t", header=None)
                        df.columns = ["chrom", "chromStart", "name"]
                        df = df.astype({"chrom": "category"})
                        df["name"] = (
                            df["name"]
                            .rolling(window=window, center=False, min_periods=1)
                            .mean()
                        )
                        df = df.iloc[:-1:window, :].reset_index()
                        df["chromEnd"] = df["chromStart"] + window - 1
                    except pd.errors.EmptyDataError:
                        if stop_on_errors:
                            raise
                        warnings.warn(f"Pandas parsing error on file {f_tmp}")
                # save output
                if "df" in locals():
                    df = df[["chrom", "chromStart", "chromEnd", "name"]]
                    _save_datframe(df, f_out, output_format)
            finally:
                if "f_tmp" in locals() and os.path.isfile(f_tmp):
                    os.remove(f_tmp)
            return f_out


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
        input_format = "parquet"
    elif input_dataframe_file.endswith(".h5") or input_dataframe_file.endswith(".hdf"):
        df = pd.read_hdf(input_dataframe_file)
        input_format = "h5"
    elif input_dataframe_file.endswith(".csv"):
        df = pd.read_csv(input_dataframe_file)
        input_format = "csv"
    elif input_dataframe_file.endswith(".tsv"):
        df = pd.read_csv(input_dataframe_file, sep="\t")
        input_format = "tsv"
    else:
        raise ValueError(
            f"""Could not interpret extension of {input_dataframe_file}\n
Should be one of "parquet", "hdf", "h5", "csv", "tsv" """
        )
    return df, input_format


def _save_datframe(df, f_out, output_format):
    if output_format == "parquet":
        df.to_parquet(f_out)
    elif output_format == "hdf" or output_format == "h5":
        df.to_hdf(f_out, "data")
    elif output_format == "csv" or output_format == "tsv":
        df.to_csv(f_out, sep="," if output_format == "csv" else "\t")
    else:
        raise ValueError(
            f"""Could not interpret output_format {output_format}\n
Should be one of "parquet", "hdf", "h5", "csv", "tsv" """
        )


def create_coverage_annotations(
    coverage_dataframe: str,
    output_annotations_file: str = None,
    coverage_intervals_dict: str = "s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_rapidQC_intervals.tsv",
    n_jobs: int = -1,
    progress_bar: bool = True,
    output_format: str = None,
):
    """

    Parameters
    ----------
    coverage_dataframe: str
        Path to pandas dataframe created by calculate_and_bin_coverage
    output_annotations_file: str
        Output file to which annotations dataframe will be written, if None (default) will be created in the same
        directory as coverage_dataframe
    coverage_intervals_dict: str
        tsv file pointing to a dataframe detailing the various intervals, currently the only instances are:
        default "s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_rapidQC_intervals.tsv"
        alternative "s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_and_whole_exome_intervals.tsv.tsv"
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
    if output_format not in [None, "parquet", "hdf", "h5", "csv", "tsv"]:
        raise ValueError(f"Unrecognized output_format {output_format}")
    multiple_inputs = not isinstance(coverage_dataframe, str) and isinstance(
        coverage_dataframe, Iterable
    )
    if multiple_inputs:  # loop over inputs
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
            )
        )
    else:
        df, input_format = _read_dataframe(coverage_dataframe)
        df = df.set_index(["chrom", "chromStart", "chromEnd"])
        if output_format is None:
            output_format = input_format
        # set output file name
        if output_annotations_file is None:
            output_annotations_file = pjoin(
                dirname(coverage_dataframe),
                ".".join(["annotations"] + basename(coverage_dataframe).split(".")),
            )
        os.makedirs(dirname(output_annotations_file), exist_ok=True)

        # fetch intervals
        coverage_intervals_dict = cloud_sync(coverage_intervals_dict)
        df_coverage_intervals = pd.read_csv(coverage_intervals_dict, sep="\t")
        df_coverage_intervals["file"] = df_coverage_intervals.apply(
            lambda x: cloud_sync(
                pjoin(dirname(coverage_intervals_dict), x["file"][2:])
            ),
            axis=1,
        )
        # start work
        with TemporaryDirectory() as tmpfile:
            # write regions bed file
            bed_file = pjoin(
                tmpfile,
                f"regions.{'.'.join(basename(coverage_dataframe).split('.')[:-1])}.bed",
            )
            df[[]].to_csv(bed_file, sep="\t", index=True)
            # create bed file per annotation
            out_intersected_beds = Parallel(n_jobs=n_jobs)(
                delayed(_intersect_intervals)(interval, bed_file)
                for interval in tqdm(
                    df_coverage_intervals["file"].values,
                    desc="Generating annotation bed files",
                    disable=not progress_bar,
                )
            )
            df_coverage_intervals["out_intersected_bed"] = out_intersected_beds
            # read annotation bed files
            df_list = [
                _read_intersected_bed(row["out_intersected_bed"], row["category"])
                for _, row in df_coverage_intervals.iterrows()
            ]
            # merge
            df_annotations = df_list[0]
            for df_tmp in tqdm(
                df_list[1:],
                desc="Merging intervals to single dataframe",
                disable=not progress_bar,
            ):
                # if df_tmp.shape[0] == 0:
                #     continue
                df_annotations = df_annotations.join(df_tmp, how="outer")
            df_annotations = df_annotations[sorted(df_annotations.columns)]
            df_annotations = df_annotations[~df_annotations.index.duplicated()]
            df_annotations = df_annotations.reindex(df.index).fillna(False)
            df_annotations = df_annotations[
                sorted(
                    df_annotations.columns,
                    key=lambda x: df_coverage_intervals.query(f"category=='{x}'")[
                        "order"
                    ].values[0],
                )
            ]
        # save
        _save_datframe(df_annotations, output_annotations_file, output_format)
        return output_annotations_file


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
    return out_intersected_bed


def _read_intersected_bed(intersected_bed, category_name):
    try:
        df = (
            pd.read_csv(intersected_bed, sep="\t", header=None,)
            .assign(**{category_name: True})
            .rename(columns={0: "chrom", 1: "chromStart", 2: "chromEnd"})
            .set_index(["chrom", "chromStart", "chromEnd"])
        )
    except pd.errors.EmptyDataError:
        df = (
            pd.DataFrame(columns=["chrom", "chromStart", "chromEnd"])
            .set_index(["chrom", "chromStart", "chromEnd"])
            .assign(**{category_name: False})
        )
    return df


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
        google_application_credentials=args_in.g,
        cloudsdk_python=args_in.cloudsdk_python,
    )
    if isinstance(output, str):
        sys.stdout.write(output)
    else:
        sys.stdout.write(os.linesep.join(output) + os.linesep)


def call_create_coverage_annotations(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    print(args)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_calculate = subparsers.add_parser(
        name="calculate",
        description="Collect coverage from an aligned bam/cram file in defined window size across a region",
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
        default="parquet",
        help="""File type of dataframe output, allowed values: "parquet" (default), "hdf", "h5", "csv", "tsv" """,
    )
    parser_calculate.add_argument(
        "--raise_errors",
        default=False,
        action="store_true",
        help="If False (default) only warnings are raised",
    )
    parser_calculate.add_argument(
        "--g",
        "--google_application_credentials",
        type=str,
        default="/home/ubuntu/.Terra_Project-d2deb59e13de.json",
        help="""Path to json file of google credentials, required for gs storage access (GOOGLE_APPLICATION_CREDENTIALS)
Tip: Check your ~/.bashrc if you are not sure what this is supposed to be""",
    )
    parser_calculate.add_argument(
        "--cloudsdk_python",
        type=str,
        default="/usr/bin/python2.7",
        help="""Path to CLOUDSDK_PYTHON env variable required for cloud (default /usr/bin/python2.7)
Tip: Check your ~/.bashrc if you are not sure what this is supposed to be""",
    )
    parser_calculate.set_defaults(func=call_calculate_and_bin_coverage)

    parser_annotate = subparsers.add_parser(
        name="annotate",
        description="Collect coverage from an aligned bam/cram file in defined window size across a region",
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
        default="s3://ultimagen-ilya-new/VariantCalling/data/coverage_intervals/coverage_chr9_rapidQC_intervals.tsv",
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
        default="parquet",
        help="""File type of dataframe output, allowed values: "parquet" (default), "hdf", "h5", "csv", "tsv" """,
    )
    parser_annotate.set_defaults(func=call_create_coverage_annotations)

    args = parser.parse_args()
    args.func(args)
