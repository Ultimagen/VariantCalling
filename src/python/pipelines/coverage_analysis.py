import pandas as pd
import numpy as np
import os
import sys
from os.path import join as pjoin, basename, dirname
from tempfile import TemporaryDirectory
import subprocess
from collections.abc import Iterable
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from glob import glob
from scipy.interpolate import interp1d

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
TMPDIR_PREFIX = "/data/tmp/tmp" if os.path.isdir("/data/tmp/") else None
DEFAULT_REFERENCE = "gs://concordance/ref_cram/hg38+ecoli+template+reference.fa"
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


def _run_shell_command(cmd, logger=logger):
    """Wrapper for running shell commands - takes care of logging and generates a GCS token if any command argument
    is a gs:// file"""
    try:
        token = (
            get_gcs_token()
            if np.any([x.startswith("gs://") for x in cmd.split()])
            else ""
        )  # only generate token if input files are on gs
        if len(token) > 0:
            logger.debug(f"gcs token generated")
        logger.debug(f"Running command:\n{cmd}")
        stdout, stderr = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env={**os.environ, **{GCS_OAUTH_TOKEN: token}},
        ).communicate()
        logger.debug(f"Finished Running command:\n{cmd}")
        logger.debug(f"stdout:\n{stdout.decode()}")
        logger.debug(f"stderr:\n{stderr.decode()}")
    except subprocess.CalledProcessError:
        warnings.warn(
            f"Error running the command:\n{cmd}\nLikely a GCS_OAUTH_TOKEN issue"
        )
        raise
    return stdout.decode(), stderr.decode()


def collect_depth(input_bam_file, output_bed_file, samtools_args=None):
    """Create a depth bed file - built on "samtools depth" but outputs a bed file

    Parameters
    ----------
    input_bam_file
    output_bed_file
    samtools_args - passed to "samtools depth"

    Returns
    -------

    """
    if samtools_args is None:
        samtools_args = []
    samtools_depth_cmd = (
        f"samtools depth {' '.join(samtools_args)} {input_bam_file}"
        + ' | awk \'{print $1"\\t"$2"\\t"($2 + 1)"\\t"$3}\''
        + f" > {output_bed_file}"
    )
    _run_shell_command(samtools_depth_cmd)
    if not os.path.isfile(output_bed_file):
        raise ValueError(
            f"file {output_bed_file} was supposed to be created but cannot be found"
        )
    return output_bed_file


def create_coverage_histogram_from_depth_file(
    input_depth_bed_file, output_tsv, region_bed_file=None
):
    """Takes an input input_depth_bed_file (create with "collect_depth") and an optional region bed file, and creates a
    coverage histogram tsv file.

    Parameters
    ----------
    input_depth_bed_file
    region_bed_file
    output_tsv

    Returns
    -------

    """
    bedtools_cmd = (
        f"bedtools intersect -wa -a {input_depth_bed_file} -b {region_bed_file}"
    )
    awk_cmd = "awk '{count[$4]++} END {for (word in count) print word, count[word]}' "
    output_cmd = f" > {output_tsv}"
    if region_bed_file is None:
        cmd = awk_cmd + input_depth_bed_file + output_cmd
    else:
        cmd = bedtools_cmd + " | " + awk_cmd + output_cmd
    _run_shell_command(cmd)
    if not os.path.isfile(output_tsv):
        raise ValueError(
            f"file {output_tsv} was supposed to be created but cannot be found"
        )
    return output_tsv


def create_binned_coverage(
    input_depth_bed_file,
    output_binned_depth_bed_file,
    lines_to_bin,
    window_size,
    generate_dataframe=False,
):
    """Takes an input input_depth_bed_file (create with "collect_depth" or previously aggregated with
    create_binned_coverage) and creates a windowed version. Optionally generates a parquet dataframe as well, with the
    same basenane as output_binned_depth_bed_file.

    Parameters
    ----------
    input_depth_bed_file
    output_binned_depth_bed_file
    lines_to_bin
        lines in bed file to group together
    window_size
        expected window size of output bed file - entries that do not match this size are due to reference discontinuty
        and are discarded. If the original file is not binned then this should be equal to lines_to_bin, otherwise it
        needs to be worked out. For example - you can apply this function twice to get a version with a window size of
        10 and another with 100.
    generate_dataframe

    Returns
    -------

    """
    cmd = (
        f"awk -vn={lines_to_bin} -vm={window_size} "
        + "'"
        + r'{sum+=$4} NR%n==1 {chr=$1;start=$2} NR%n==0 {end=$3} (NR%n==0) && (end-start==m) {print chr "\t" start "\t" end "\t" sum/n} NR%n==0 {sum=0}'
        + "'"
        + f" {input_depth_bed_file} > {output_binned_depth_bed_file}"
    )
    _run_shell_command(cmd)
    if generate_dataframe:
        output_parquet = (
            output_binned_depth_bed_file[:-4]
            if output_binned_depth_bed_file.endswith(".bed")
            or output_binned_depth_bed_file.endswith(".tsv")
            else output_binned_depth_bed_file
        ) + ".parquet"
        pd.read_csv(
            output_binned_depth_bed_file,
            sep="\t",
            header=None,
            names=[CHROM, CHROM_START, CHROM_END, COVERAGE],
        ).astype({CHROM: CHROM_DTYPE}).to_parquet(output_parquet)
        return output_parquet


def _intervals_to_bed(input_intervals, output_bed_file=None):
    """convert picard intervals to bed

    Parameters
    ----------
    input_intervals
    output_bed_file
        If None (default), the input file with a modified extension is used

    Returns
    -------

    """
    if output_bed_file is None:
        output_bed_file = input_intervals
        if output_bed_file.endswith(".interval"):
            output_bed_file = output_bed_file[: -len(".interval")]
        if output_bed_file.endswith(".interval_list"):
            output_bed_file = output_bed_file[: -len(".interval_list")]
        output_bed_file += ".bed"
    cmd_create_bed = (
        f"picard IntervalListToBed INPUT={input_intervals} OUTPUT={output_bed_file}"
    )
    _run_shell_command(cmd_create_bed)
    if not os.path.isfile(output_bed_file):
        raise ValueError(
            f"file {output_bed_file} was supposed to be created but cannot be found"
        )
    return output_bed_file


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
        coverage_intervals_dict_local = cloud_sync(coverage_intervals_dict)
        df_coverage_intervals = pd.read_csv(coverage_intervals_dict_local, sep=sep)
        df_coverage_intervals["file"] = df_coverage_intervals.apply(
            lambda x: _intervals_to_bed(
                cloud_sync(pjoin(dirname(coverage_intervals_dict), x["file"][2:]))
            ),
            axis=1,
        )
    elif isinstance(coverage_intervals_dict, dict):
        df_coverage_intervals = pd.DataFrame.from_dict(
            coverage_intervals_dict, orient="index"
        ).reset_index()
        df_coverage_intervals.columns = ["category", "file"]
        df_coverage_intervals["file"] = df_coverage_intervals.apply(
            lambda x: _intervals_to_bed(cloud_sync(x["file"])), axis=1,
        )
    else:
        raise ValueError(f"Invalid input {coverage_intervals_dict}")

    if "order" not in df_coverage_intervals:
        df_coverage_intervals = df_coverage_intervals.assign(
            order=range(df_coverage_intervals.shape[0])
        )

    return df_coverage_intervals


def _get_output_file_name(
    f_in, f_out, min_bq, min_mapq, min_read_length, region, window, output_format,
):
    if f_out is None or "." not in basename(f_out):
        local_f_in = cloud_sync(f_in, dry_run=True)
        if f_out is None:
            local_dirname = dirname(local_f_in)
        else:
            local_dirname = f_out
        extra_extensions = f".q{min_bq}.Q{min_mapq}.l{min_read_length}"
        out_basename = pjoin(local_dirname, basename(f_in).split(".")[0])
        region_str = "" if region is None else "." + region.replace(":", "_")
        f_out = (
            f"{out_basename}{region_str}{extra_extensions}.w{window}.{output_format}"
        )
    return f_out


def generate_stats_from_histogram(
    val_count,
    q=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.95]),
    out_path=None,
    verbose=True,
):
    if isinstance(val_count, str) and os.path.isfile(val_count):
        val_count = pd.read_hdf(val_count, key="histogram")
    df_precentiles = pd.concat(
        (
            val_count.apply(lambda x: interp1d(np.cumsum(x), val_count.index, fill_value="extrapolate")(q)),
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


def generate_coverage_boxplot(
    df_percentiles, color_group=None, out_path=None, title=""
):
    if out_path is not None:
        mpl.use("Agg")
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


def _check_chr_in_file_name(filename):
    for x in basename(filename).split("."):
        if x.startswith("chr"):
            return x
    return None


def plot_coverage_profile(
    input_depth_files,
    cyto_band_file="gs://ultimagen-sarah/ucsc_tables/CytoBandIdeo",
    reference_gaps_file="gs://ultimagen-sarah/ucsc_tables/gap.txt",
    title="",
    sub_title="",
    y_max=3,
    out_path=None,
):
    if out_path is not None:
        mpl.use("Agg")
    df_acen = (
        pd.read_csv(
            cloud_sync(cyto_band_file),
            sep="\t",
            header=None,
            names=["chrom", "chromStart", "chromEnd", "type"],
            usecols=[0, 1, 2, 4],
            comment="#",
        )
        .query("type == 'acen'")
        .drop(columns=["type"])
        .set_index("chrom")
    )
    df_gaps = pd.read_csv(
        cloud_sync(reference_gaps_file),
        sep="\t",
        header=None,
        names=["chrom", "chromStart", "chromEnd"],
        usecols=range(1, 4),
        comment="#",
    ).set_index("chrom")

    median_coverage = np.median(
        [pd.read_parquet(f)["coverage"].median() for f in input_depth_files.values()]
    )
    N = len(input_depth_files)

    fig, axs = plt.subplots(
        np.ceil(N / 2).astype(int),
        2,
        figsize=(28, np.ceil(N / 2).astype(int) * 3),
        sharey="all",
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.01)
    suptitle = fig.suptitle(
        f"""Coverage profile (normalized to median) {title}
    Median coverage = {median_coverage:.1f}{sub_title}""",
        # Blue - coverage profile, Red - reference gaps, Green - centromeres""",
        y=0.96,
    )
    for ax, r in zip(axs.flatten(), input_depth_files.keys()):
        plt.sca(ax)

        plt.title(r, fontsize=18)
        df = pd.read_parquet(input_depth_files[r])
        x = (df["chromStart"] + df["chromEnd"]) / (2e6)
        plt.plot(
            x, df["coverage"] / median_coverage, label="coverage profile", zorder=1,
        )
        for j, (_, row) in enumerate(df_gaps.loc[[r]].iterrows()):
            plt.fill_betweenx(
                [0, y_max + 1],
                row["chromStart"] / 1e6,
                row["chromEnd"] / 1e6,
                facecolor="red",
                alpha=0.9,
                label="reference gaps" if j == 0 else None,
                zorder=2,
            )
        for j, (_, row) in enumerate(df_acen.loc[[r]].iterrows()):
            plt.fill_betweenx(
                [0, y_max + 1],
                row["chromStart"] / 1e6,
                row["chromEnd"] / 1e6,
                facecolor="green",
                label="centromeres" if j == 0 else None,
                alpha=0.9,
                zorder=2,
            )
        plt.xlim(x.min(), x.max())

    for ax in axs[-1, :] if len(axs.shape) > 1 else axs:
        xlabel = ax.set_xlabel("Position [Mb]")
    leg = axs.flatten()[0].legend(bbox_to_anchor=[1, 2])
    ax.set_ylim(0, y_max)

    if out_path is not None:
        if "." not in basename(out_path):  # interpret as directory
            out_path = pjoin(out_path, "coverage_boxplot.png")
        fig.savefig(
            out_path,
            dpi=300,
            bbox_extra_artists=[suptitle, leg, xlabel],
            bbox_inches="tight",
        )

    return fig


def run_full_coverage_analysis(
    bam_file: str,
    out_path: str,
    coverage_intervals_dict: str = DEFAULT_INTERVALS,
    regions: str = None,
    windows: int = None,
    min_bq: int = 0,
    min_mapq: int = 0,
    min_read_length: int = 0,
    ref_fasta: str = DEFAULT_REFERENCE,
    n_jobs: int = -1,
    progress_bar: bool = True,
    verbose=False,
    cyto_band_file="gs://ultimagen-sarah/ucsc_tables/CytoBandIdeo",
    reference_gaps_file="gs://ultimagen-sarah/ucsc_tables/gap.txt",
):
    # check inputs
    os.makedirs(out_path, exist_ok=True)
    if windows is None:
        windows = [100, 1000, 10000, 100000]
    w0 = 1
    for w in windows:
        if w % w0 != 0:
            raise ValueError(
                f"consecutive window sizes must divide by each other, got {windows}"
            )
    if regions is None:
        regions = CHROM_DTYPE.categories
    elif isinstance(regions, str) or not isinstance(regions, Iterable):
        regions = [regions]
    bam_file_name = basename(bam_file).split(".")[0]
    params_filename_suffix = f"q{min_bq}.Q{min_mapq}.l{min_read_length}"

    # set samtools arguments
    is_cram = bam_file.endswith(CRAM_EXT)
    ref_str = ""
    if is_cram:
        ref_fasta = cloud_sync(ref_fasta)
        ref_str = f"--reference {ref_fasta}"

    samtools_depth_args = [
        "-a",
        "-J",
        ref_str,
        f"-q {min_bq}",
        f"-Q {min_mapq}",
        f"-l {min_read_length}",
    ]
    # collect depth
    out_depth_files = Parallel(n_jobs=n_jobs)(
        delayed(collect_depth)(
            bam_file,
            _get_output_file_name(
                bam_file,
                out_path,
                min_bq,
                min_mapq,
                min_read_length,
                region,
                window=1,
                output_format="depth",
            ),
            samtools_args=" ".join(
                samtools_depth_args + [f"-r {region}" if region is not None else ""]
            ).split(),
        )
        for region in tqdm(
            regions,
            disable=not progress_bar,
            desc="Creating depth files using samtools",
        )
    )

    # collect coverage in intervals
    if coverage_intervals_dict is not None:
        if verbose:
            logger.debug(
                f"Collecting coverage in genomic intervals from {coverage_intervals_dict}"
            )
        df_coverage_intervals = _create_coverage_intervals_dataframe(
            coverage_intervals_dict
        )
        # the next segment run "create_coverage_histogram_from_depth_file" for all the combinations of region and
        # interval. it makes sure that if the filenames are annotated with "chrX" somewhere that the chrom is the same
        # between the region and bed interval (most bed file are for chr9 only, no point in running them with the other
        # depth files)
        with TemporaryDirectory(prefix=pjoin(out_path, "tmp")) as tmp_basedir:
            tmpdir = dict()
            for region_bed_file in df_coverage_intervals["file"].values:
                tmpdir[region_bed_file] = pjoin(
                    tmp_basedir, f"tmp.{'.'.join(basename(region_bed_file).split('.')[:-1])}"
                )
                os.makedirs(tmpdir[region_bed_file], exist_ok=True)

            Parallel(n_jobs=n_jobs)(
                delayed(create_coverage_histogram_from_depth_file)(
                    input_depth_bed_file,
                    pjoin(
                        tmpdir[region_bed_file],
                        f"{basename(input_depth_bed_file).split('.depth')[0]}.coverage_histogram.tsv",
                    ),
                    region_bed_file,
                )
                for input_depth_bed_file, region_bed_file in [
                    (x, y)
                    for x, y in itertools.product(
                        out_depth_files, df_coverage_intervals["file"].values
                    )
                    if (_check_chr_in_file_name(x) == _check_chr_in_file_name(y))
                    or (_check_chr_in_file_name(y) is None)
                ]
            )

            # remove regions with not outputs
            keys_to_remove = list()
            for original_bed_file, tmp_region_dir in tmpdir.items():
                if len(glob(pjoin(tmp_region_dir, "*.tsv"))) == 0:
                    keys_to_remove.append(original_bed_file)
            for k in keys_to_remove:
                tmpdir.pop(k)

            df_coverage_histogram = (
                pd.concat(
                    (
                        pd.concat(
                            (
                                pd.read_csv(
                                    tsv_file,
                                    sep=" ",
                                    header=None,
                                    names=["coverage", "count"],
                                )
                                for tsv_file in glob(pjoin(tmp_region_dir, "*.tsv"))
                            )
                        )
                        .groupby("coverage")
                        .sum()
                        .rename(
                            columns={
                                "count": df_coverage_intervals[
                                    df_coverage_intervals["file"] == original_bed_file
                                ]["category"].values[0]
                            }
                        )
                        for original_bed_file, tmp_region_dir in tmpdir.items()
                    ),
                    axis=1,
                )
                .fillna(0)
                .astype(int)
            )
        # save coverage in intervals and create boxplot
        df_percentiles, df_stats = generate_stats_from_histogram(
            df_coverage_histogram / df_coverage_histogram.sum()
        )
        if verbose:
            logger.debug(f"Saving data")
        if "." in basename(out_path):
            coverage_stats_dataframes = out_path
        else:
            os.makedirs(out_path, exist_ok=True)
            coverage_stats_dataframes = pjoin(
                out_path, f"{bam_file_name}.coverage_stats.{params_filename_suffix}.h5"
            )
        if verbose:
            logger.debug(f"Saving dataframes to {coverage_stats_dataframes}")
        df_stats.to_hdf(coverage_stats_dataframes, key="stats", mode="a")
        df_percentiles.to_hdf(coverage_stats_dataframes, key="percentiles", mode="a")
        df_coverage_histogram.to_hdf(
            coverage_stats_dataframes, key="histogram", mode="a"
        )

        generate_coverage_boxplot(
            df_percentiles,
            coverage_intervals_dict,
            out_path=pjoin(out_path, f"coverage_boxplot.{params_filename_suffix}.png"),
            title=bam_file_name,
        )

    # create binned dataframes
    if windows is not None:
        if verbose:
            logger.debug(f"creating binned dataframes for window sizes: {windows}")
        depth_files_to_process = out_depth_files
        w0 = 1
        for j, w in enumerate(windows):
            Parallel(n_jobs=n_jobs)(
                delayed(create_binned_coverage)(
                    input_depth_bed_file=depth_file,
                    output_binned_depth_bed_file=depth_file.split(".w")[0]
                    + f".w{w}.depth",
                    lines_to_bin=w // w0,
                    window_size=w,
                    generate_dataframe=True,
                )
                for depth_file in tqdm(
                    depth_files_to_process,
                    disable=not progress_bar,
                    desc=f"Binning depth files ({j+1}/{len(windows)}, w={w})",
                )
            )

            depth_files_to_process = [
                depth_file.split(".w")[0] + f".w{w}.depth"
                for depth_file in out_depth_files
            ]

            # plot coverage profile
            if w >= 1000:  # below that the graph is useless
                plot_coverage_profile(
                    input_depth_files={
                        r: f + ".parquet"
                        for r, f in zip(regions, depth_files_to_process)
                        if r != "chrM"
                    },
                    cyto_band_file=cyto_band_file,
                    reference_gaps_file=reference_gaps_file,
                    title=bam_file_name,
                    sub_title=f", Window = {w if w < 1000 else str(w // 1000) + 'k'}b, MapQ >= {min_mapq}",
                    out_path=pjoin(
                        out_path,
                        f"{bam_file_name}.coverage_profile.w{w if w < 1000 else str(w // 1000) + 'k'}b.{params_filename_suffix}.png",
                    ),
                )
            # set new parameters so that the next window size is a processing of the binned file and not the original
            w0 = w


def call_run_full_coverage_analysis(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    run_full_coverage_analysis(
        bam_file=args_in.input,
        out_path=args_in.output,
        coverage_intervals_dict=args_in.coverage_intervals,
        regions=args_in.region,
        windows=args_in.windows,
        min_bq=args_in.q,
        min_mapq=args_in.Q,
        min_read_length=args_in.l,
        ref_fasta=args_in.reference,
        n_jobs=args_in.jobs,
        progress_bar=not args_in.no_progress_bar,
    )
    sys.stdout.write("DONE" + os.linesep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_full_analysis = subparsers.add_parser(
        name="full_analysis",
        description="""Run full coverage analysis of an aligned bam/cram file""",
    )
    parser_full_analysis.add_argument(
        "-i", "--input", type=str, help="input bam or cram file ",
    )
    parser_full_analysis.add_argument(
        "-o",
        "--output",
        type=str,
        help="""Path to which output dataframe will be written, will be created if it does not exist""",
    )
    parser_full_analysis.add_argument(
        "-c",
        "--coverage_intervals",
        type=str,
        default=DEFAULT_INTERVALS,
        help=f"""tsv file pointing to a dataframe detailing the various intervals
default {DEFAULT_INTERVALS}""",
    )
    parser_full_analysis.add_argument(
        "-r",
        "--region",
        type=str,
        nargs="*",
        default=CHROM_DTYPE.categories,
        help=f"""Genomic region in samtools format - default is {CHROM_DTYPE.categories.values} """,
    )
    parser_full_analysis.add_argument(
        "-w",
        "--windows",
        type=int,
        default=None,
        help="Number of base pairs to bin coverage by (leave blank for default [100, 1000, 10000, 100000])",
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
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE,
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
    parser_full_analysis.set_defaults(func=call_run_full_coverage_analysis)

    args = parser.parse_args()
    args.func(args)
