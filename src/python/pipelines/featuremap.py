import os
import sys
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import pysam
import pyfaidx
import itertools
import argparse
import gzip
import pyBigWig as pbw
from os.path import dirname, basename, join as pjoin
from collections.abc import Iterable
from scipy.interpolate import interp1d
if __name__ == "__main__":
    import pathmagic

import logging

logger = logging.getLogger("featuremap")
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

from python.utils import revcomp, generateKeyFromSequence
from python.modules.variant_annotation import get_motif_around
from python.auxiliary.format import (
    CHROM_DTYPE,
    CYCLE_SKIP_DTYPE,
    CYCLE_SKIP_STATUS,
    CYCLE_SKIP,
    POSSIBLE_CYCLE_SKIP,
    NON_CYCLE_SKIP,
)

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


def _collect_coverage_per_motif(
    chrom: str, depth_file: str, size: int = 5, N: int = 100,
):
    """
    Collect coverage per motif from given input file from a single chromosome

    Parameters
    ----------
    chrom
        sequence of the chromosome that matches the input depth_file, all upper case - as obtained by
        pyfaidx.Fasta(reference_fasta)[chr_str][:].seq.upper()
    depth_file
        depth file (generated by coverage_analysis:run_full_coverage_analysis in this repo)
    size
        maximal motif size, deault 5 - all motifs below that number until 0 are also calculated
    N
        process 1 in every N positions - makes the code run faster the larger N is (default 100).

    Returns
        dataframe with column f"motif_{size}" (motif in the reference), and "count" (how many times it was counted)
    -------

    """

    counter = defaultdict(lambda: 0)
    search = re.compile(r"[^ACGTacgt.]").search
    if not depth_file.endswith("bw"):
        open_func = gzip.open if depth_file.endswith(".gz") else open
        with open_func(depth_file) as f:
            for j, line in enumerate(f):
                if j % N != 0:
                    continue
                if isinstance(line, bytes):
                    line = line.decode()
                line = line.strip()
                spl = line.split("\t")
                pos = int(spl[1])
                cov = int(spl[3])
                if pos < 2 * size or cov == 0:
                    continue
                seq = chrom[pos - size - 1:pos + size]

                if not bool(search(seq)):  # no letters other than ACGT
                    counter[seq] += cov

    else:  # case of bigWig - we fetch by region
        CHUNK_SIZE = 1000000
        with pbw.open(depth_file) as f:
            assert len(list(f.chroms().keys())) == 1, "Expected single chromosome per bw"
            chrom_name = list(f.chroms().keys())[0]
            chrom_len = f.chroms()[chrom_name]
            start_points = np.arange(0, chrom_len, CHUNK_SIZE).astype(np.int)
            for s in start_points:
                vals = f.values(chrom_name, s, min(s+CHUNK_SIZE, chrom_len))
                vals = vals[::N]
                poss = np.arange(s, min(s+CHUNK_SIZE, chrom_len))[::N]
                for i, pos in enumerate(poss):
                    cov = vals[i]
                    if pos < 2 * size or cov == 0 or np.isnan(cov):
                        continue
                    seq = chrom[pos - size - 1:pos + size]

                    if not bool(search(seq)):  # no letters other than ACGT
                        counter[seq] += cov

    df = (
        pd.DataFrame(counter.values(), index=counter.keys())
        .reset_index()
        .rename(columns={"index": f"motif_{size}", 0: "count"})
    )
    

    # edge case of empty dataframe

    if 'count' not in df.columns :
        df['count'] = []
    return df


def collect_coverage_per_motif(
    depth_files,
    reference_fasta: str,
    outfile: str = None,
    show_stats: bool = False,
    n_jobs: int = -1,
    size: int = 5,
    N: int = 100,
):
    """
    Collect coverage per motif from given input files, output is a set of dataframes with motif of a given size and
    columns "count", which is the number of occurences of this motif in the sampled data, and "coverage" which is an
    extrapolation to the full data set

    Parameters
    ----------
    depth_files
        dictionary of depth files in BW or bed format (generated by coverage_analysis:run_full_coverage_analysis in this repo), where keys
        are the chromosomes and values are the files (window size 1)
        alternatively, this can be an Iterable of files, but the filenames must follow the convention where each file
        contains reads from one chromosome only, and the file basename contains the chromosome name surrounded by dots,
        i.e. /path/to/file/filename.chr1.bw
    reference_fasta

    outfile
        output file to save to, hdf format - each motif of size num is save under the key "motif_{num}", if None
        (default) output is not saved
    show_stats
        print statistics for each calculated motif, default False
    n_jobs
        number of processes to run in parallel, -1 (default) for maximal possible number
    size
        maximal motif size, deault 5 - all motifs below that number until 0 are also calculated
    N
        process 1 in every N positions - makes the code run faster the larger N is (default 100). in the output
        dataframe df, df['coverage'] = df['count'] * N

    Returns
    -------
    motif coverage dataframe

    """
 
    if not isinstance(depth_files, dict):
        if not isinstance(depth_files, Iterable):
            raise ValueError(f"Expected dictionary or Iterable, got:\n{depth_files}")

        def _extract_chrom(fname):
            for x in basename(fname).split("."):
                if x.startswith("chr"):
                    return x
            raise ValueError(f"Could not figure out chromosome of the file {fname}")

        # convert to dictionary
        depth_files = {_extract_chrom(fname): fname for fname in depth_files}

    df = pd.concat(
        Parallel(n_jobs=n_jobs)(
            delayed(_collect_coverage_per_motif)(
                chrom=pyfaidx.Fasta(reference_fasta)[chr_str][:].seq.upper(),
                depth_file=depth_file,
                size=size,
                N=N,
            )
            for chr_str, depth_file in depth_files.items()
        )
    )
    df = df.groupby(f"motif_{size}").sum()
    df = df.reset_index()
    for j in list(range(size))[::-1]:
        df[f"motif_{j}"] = df[f"motif_{j + 1}"].str.slice(1, -1)

    df = df[[f"motif_{j}" for j in range(size + 1)] + ["count"]]
    df["coverage"] = df["count"] * N
    if outfile is not None:
        for j in range(size + 1):
            df.groupby(f"motif_{j}").agg({"count": "sum", "coverage": "sum"}).to_hdf(
                outfile, key=f"motif_{j}"
            )
    if show_stats:
        print(
            pd.concat(
                (
                    df.groupby(f"motif_{j}")
                    .agg({"count": "sum"})
                    .describe()
                    .rename(columns={"count": f"motif_{j}"})
                    for j in range(size + 1)
                ),
                axis=1,
            )
        )
    return df


def featuremap_to_dataframe(
    featuremap_vcf: str,
    output_file: str = None,
    reference_fasta: str = None,
    motif_length: int = 4,
    report_read_orientation: bool = True,
    x_fields: list = None,
    show_progress_bar: bool = False,
    flow_order: str = "TGCA",
):
    """
    Converts featuremap in vcf format to dataframe
    if reference_fasta, annotates with motifs of length "motif_length"
    if flow_order is also given, annotates cycle skip status per entry

    Parameters
    ----------
    featuremap_vcf
        featuremap file generated by "gatk FeatureMap"
    output_file
        file path to save
    reference_fasta
        reference genome used to generate the bam that the featuremap was generated from, if not None (default) the
        entries in the featuremap are annorated for motifs with the length of the next parameters from either side
    motif_length
        default 4
    report_read_orientation
        featuremap entries are reported for the sense strand regardless of read diretion. If True (default), the ref and
        alt columns are reverse complemented for reverse strand reads (also motif if calculated).
    x_fields
        fields to extract from featuremap, if default (None) those are extracted:
        "X-CIGAR", "X-EDIST", "X-FC1", "X-FC2", "X-FILTERED-COUNT", "X-FLAGS", "X-LENGTH", "X-MAPQ", "X-READ-COUNT",
        "X-RN", "X-SCORE", "rq",
    show_progress_bar
        displays tqdm progress bar of number of lines read (not in percent)
    flow_order
        flow order

    Returns
    -------

    """
    if x_fields is None:
        x_fields = [
            "X-CIGAR",
            "X-EDIST",
            "X-FC1",
            "X-FC2",
            "X-FILTERED-COUNT",
            "X-FLAGS",
            "X-LENGTH",
            "X-MAPQ",
            "X-READ-COUNT",
            "X-RN",
            "X-SCORE",
            "rq",
        ]

    with pysam.VariantFile(featuremap_vcf) as f:
        vfi = map(
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
            f,
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

    if report_read_orientation:
        is_reverse = ~(df["X-FLAGS"] & 16).astype(bool)
        for c in ["ref", "alt"]:  # reverse value to match the read direction
            df[c] = df[c].where(is_reverse, df[c].apply(revcomp))

    if reference_fasta is not None:
        df = (
            get_motif_around(df.assign(indel=False), motif_length, reference_fasta)
            .drop(columns=["indel"])
            .astype({"left_motif": str, "right_motif": str})
        )

        if report_read_orientation:
            left_motif_reverse = df["left_motif"].apply(revcomp)
            right_motif_reverse = df["right_motif"].apply(revcomp)
            df["left_motif"] = df["left_motif"].where(is_reverse, right_motif_reverse)
            df["right_motif"] = df["right_motif"].where(is_reverse, left_motif_reverse)

        df["ref_motif"] = (
            df["left_motif"].str.slice(-1)
            + df["ref"]
            + df["right_motif"].str.slice(0, 1)
        )
        df["alt_motif"] = (
            df["left_motif"].str.slice(-1)
            + df["alt"]
            + df["right_motif"].str.slice(0, 1)
        )
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

    df = df.set_index(["chrom", "pos"]).sort_index()
    if output_file is None:
        if featuremap_vcf.endswith(".vcf.gz"):
            output_file = featuremap_vcf[: -len(".vcf.gz")] + ".parquet"
        else:
            output_file = featuremap_vcf + ".parquet"
    df.to_parquet(output_file)
    return df


def determine_cycle_skip_status(ref: str, alt: str, flow_order: str):
    """return the cycle skip status, expects input of ref and alt sequences composed of 3 bases where only the 2nd base
    differs"""
    if (
        len(ref) != 3
        or len(alt) != 3
        or ref[0] != alt[0]
        or ref[2] != alt[2]
        or ref == alt
    ):
        raise ValueError(
            f"""Invalid inputs ref={ref}, alt={alt}
expecting input of ref and alt sequences composed of 3 bases where only the 2nd base differs"""
        )
    ref_key = np.trim_zeros(generateKeyFromSequence(ref, flow_order), "f")
    alt_key = np.trim_zeros(generateKeyFromSequence(alt, flow_order), "f")
    if len(ref_key) != len(alt_key):
        return CYCLE_SKIP
    else:
        for r, a in zip(ref_key, alt_key):
            if (r != a) and ((r == 0) or (a == 0)):
                return POSSIBLE_CYCLE_SKIP
        return NON_CYCLE_SKIP


def get_cycle_skip_dataframe(flow_order: str = "TGCA"):
    ind = pd.MultiIndex.from_tuples(
        [
            x
            for x in itertools.product(
                ["".join(x) for x in itertools.product(["A", "C", "G", "T"], repeat=3)],
                ["A", "C", "G", "T"],
            )
            if x[0][1] != x[1]
        ],
        names=["ref_motif", "alt_motif"],
    )
    df_cskp = pd.DataFrame(index=ind).reset_index()
    df_cskp["alt_motif"] = (
        df_cskp["ref_motif"].str.slice(0, 1)
        + df_cskp["alt_motif"]
        + df_cskp["ref_motif"].str.slice(-1)
    )
    df_cskp[CYCLE_SKIP_STATUS] = df_cskp.apply(
        lambda row: determine_cycle_skip_status(
            row["ref_motif"], row["alt_motif"], flow_order
        ),
        axis=1,
    ).astype(CYCLE_SKIP_DTYPE)
    return df_cskp.set_index(["ref_motif", "alt_motif"])


def merge_featuremap_dataframes(dataframes: list, outfile: str, n_jobs: int = 1):
    df = pd.concat(
        Parallel(n_jobs=n_jobs)(delayed(pd.read_parquet)(f) for f in dataframes)
    )
    df = df.sort_index()
    df.to_parquet(outfile)
    return df


def calculate_snp_error_rate(
    single_substitution_featuremap,
    coverage_stats,
    depth_data,
    out_path: str,
    out_basename: str = "",
    xscore_thresholds: list = None,
    reference_fasta: str = None,
    flow_order: str = "TGCA",
    featuremap_chrom: str = None,
    n_jobs=-1,
    N=100,
):
    # init
    if xscore_thresholds is None:
        xscore_thresholds = [3, 5, 10]
    assert len(xscore_thresholds) == 3
    min_xscore = xscore_thresholds[0]

    os.makedirs(out_path, exist_ok=True)
    if len(out_basename) > 0 and not out_basename.endswith("."):
        out_basename += "."
    out_coverage_per_motif = pjoin(out_path, f"{out_basename}coverage_per_motif.h5")
    out_snp_rate = pjoin(out_path, f"{out_basename}snp_error_rate.h5")
    out_snp_rate_plots = {
        th: pjoin(out_path, f"{out_basename}snp_error_rate_threshold{th}.png")
        for th in xscore_thresholds
    }

    # read coverage stats and derive coverage range
    if isinstance(coverage_stats, pd.Series):
        df_coverage_stats = coverage_stats
    else:
        logger.debug(f"Reading input coverage stats from {coverage_stats}")
        df_coverage_stats = pd.read_hdf(coverage_stats, key="histogram",)["Genome"]
    f = interp1d(
        (df_coverage_stats.cumsum() / df_coverage_stats.sum()).values,
        df_coverage_stats.index.values,
    )
    min_coverage = min(
        20, np.round(f(0.5)).astype(int)
    )  # the lower between 20 or the median value
    max_coverage = np.round(f(0.95)).astype(int)

    x = df_coverage_stats[
        (df_coverage_stats.index.values >= min_coverage)
        & (df_coverage_stats.index.values <= max_coverage)
    ]
    coverage_correction_factor = (x * x.index.values).sum() / (
        df_coverage_stats * df_coverage_stats.index
    ).sum()
    logger.debug(
        f"Coverage range {min_coverage}-{max_coverage}x, spanning {coverage_correction_factor:.0%} of the data"
    )
    # generate or read coverage per motif
    if isinstance(depth_data, pd.DataFrame):
        df_coverage = depth_data
    else:
        if reference_fasta is None:
            raise ValueError("Reference fasta must be provided if input is depth files")
        logger.debug(f"Generating coverage per motif")
        _ = collect_coverage_per_motif(
            depth_data,
            reference_fasta,
            outfile=out_coverage_per_motif,
            show_stats=False,
            n_jobs=n_jobs,
            size=2,
            N=N,
        )
        df_coverage = pd.read_hdf(out_coverage_per_motif, "motif_2")
    df_coverage.index.name = "ref_motif2"
    # read featuremap
    if isinstance(single_substitution_featuremap, pd.DataFrame):
        df = single_substitution_featuremap
        df = df[
            (df["X-SCORE"] >= min_xscore)
            & (df["X-READ-COUNT"] >= min_coverage)
            & (df["X-READ-COUNT"] <= max_coverage)
        ]
    else:
        logger.debug(
            f"Reading featuremap dataframe from {single_substitution_featuremap}"
        )
        df = pd.read_parquet(
            single_substitution_featuremap,
            filters=[
                ("X-SCORE", ">=", min_xscore),
                ("X-READ-COUNT", ">=", min_coverage),
                ("X-READ-COUNT", "<=", max_coverage),
            ],
        )
        if featuremap_chrom is not None:
            logger.debug(f"using only data in {featuremap_chrom}")
            df = df.loc[featuremap_chrom]
    # calculate read filtration ratio in featuremap
    read_filter_correction_factor = (df["X-FILTERED-COUNT"] + 1).sum() / df[
        "X-READ-COUNT"
    ].sum()
    # process 2nd order motifs
    logger.debug(f"Processing motifs")
    l = df["left_motif"].astype(str).str.slice(2,)
    r = df["right_motif"].astype(str).str.slice(0, -2)

    df["ref_motif2"] = l + df["ref"].astype(str) + r
    df["alt_motif2"] = l + df["alt"].astype(str) + r
    logger.debug(f"Grouping by 2nd order motif")
    df_motifs_2 = df.groupby(["ref_motif2", "alt_motif2"]).agg(
        {
            **{x: "first" for x in ["ref", "alt", "ref_motif", "alt_motif"]},
            **{
                "X-SCORE": [
                    lambda a: np.sum(a >= xscore_thresholds[0]).astype(int),
                    lambda a: np.sum(a >= xscore_thresholds[1]).astype(int),
                    lambda a: np.sum(a >= xscore_thresholds[2]).astype(int),
                ]
            },
        }
    )
    df_motifs_2 = df_motifs_2.dropna(how="all")

    df_motifs_2.columns = ["ref", "alt", "ref_motif", "alt_motif"] + [
        f"snp_count_thresh{th}" for th in xscore_thresholds
    ]
    logger.debug(f"Annotating cycle skip")
    df_motifs_2 = (
        df_motifs_2.reset_index()
        .set_index(["ref_motif", "alt_motif"])
        .join(get_cycle_skip_dataframe(flow_order=flow_order))
    )

    df_motifs_2 = (
        df_motifs_2.reset_index()
        .set_index("ref_motif2")
        .join(df_coverage[["coverage"]])
    )

    # process 1st order motifs
    logger.debug(f"Creating 1st order motif data")
    df_motifs_1 = (
        df_motifs_2.groupby(["ref_motif", "alt_motif"])
        .agg(
            {
                **{"ref": "first", "alt": "first", "cycle_skip_status": "first"},
                **{
                    c: "sum"
                    for c in df_motifs_2.columns
                    if "snp_count" in c or "coverage" in c
                },
            }
        )
        .dropna(how="all")
    )
    # process 0 order motifs
    logger.debug(f"Creating 0 order motif data")
    df_motifs_0 = (
        df_motifs_1.groupby(["ref", "alt"])
        .agg(
            {
                c: "sum"
                for c in df_motifs_1.columns
                if "snp_count" in c or "coverage" in c
            }
        )
        .dropna(how="all")
    )

    logger.debug(f"Setting non-cycle skip motifs at X-SCORE>=6 to NaN")
    for th in xscore_thresholds:
        if th >= 6:
            df_motifs_2.loc[:, f"snp_count_thresh{th}"] = df_motifs_2[
                f"snp_count_thresh{th}"
            ].where(df_motifs_2[CYCLE_SKIP_STATUS] == CYCLE_SKIP)
            df_motifs_1.loc[:, f"snp_count_thresh{th}"] = df_motifs_1[
                f"snp_count_thresh{th}"
            ].where(df_motifs_1[CYCLE_SKIP_STATUS] == CYCLE_SKIP)

    logger.debug(f"Assigning error rates")
    for df_tmp in [df_motifs_0, df_motifs_1, df_motifs_2]:
        for th in xscore_thresholds:
            df_tmp.loc[:, f"error_rate_thresh{th}"] = df_tmp[
                f"snp_count_thresh{th}"
            ] / (
                df_tmp["coverage"]
                * read_filter_correction_factor
                * coverage_correction_factor
            )

    # save
    logger.debug(f"Saving to {out_snp_rate}")
    df_motifs_2 = df_motifs_2.reset_index().astype(
        {
            c: "category"
            for c in [
                "ref",
                "alt",
                "ref_motif",
                "alt_motif",
                "ref_motif2",
                "alt_motif2",
            ]
        }
    )
    df_motifs_1 = df_motifs_1.reset_index().astype(
        {c: "category" for c in ["ref", "alt", "ref_motif", "alt_motif",]}
    )
    df_motifs_0 = df_motifs_0.reset_index().astype(
        {c: "category" for c in ["ref", "alt"]}
    )
    df_motifs_0.to_hdf(out_snp_rate, key="motif_0", mode="w", format="table")
    df_motifs_1.to_hdf(out_snp_rate, key="motif_1", mode="a", format="table")
    df_motifs_2.to_hdf(out_snp_rate, key="motif_2", mode="a", format="table")

    # generate plots
    for th in xscore_thresholds:
        logger.debug(f"Generating plot for X-SCORE>={th}")
        error_rate_column = f"error_rate_thresh{th}"
        snp_count_column = f"snp_count_thresh{th}"
        _plot_snp_error_rate(
            df_motifs_1.rename(
                columns={error_rate_column: "error_rate", snp_count_column: "snp_count"}
            ).set_index(["ref_motif", "alt_motif"]),
            out_filename=out_snp_rate_plots[th],
            title=f"{out_basename}\nLog-likelihood threshold = {th}",
            left_bbox_text=f"Coverage range {min_coverage}-{max_coverage}x\nspanning {coverage_correction_factor:.0%} of the data",
        )


def _plot_snp_error_rate(
    df_motifs: pd.DataFrame,
    out_filename: str = None,
    title: str = "",
    left_bbox_text: str = None,
):
    # init
    error_rate_column = f"error_rate"
    snp_count_column = f"snp_count"
    assert snp_count_column in df_motifs
    assert error_rate_column in df_motifs
    if out_filename is None:
        matplotlib.use("Qt5Agg")
    else:
        matplotlib.use("Agg")  # non interactive
    w = 0.3
    scale = "log"

    # create matched forward-reverse dataframe
    df_motifs["ord"] = df_motifs.index.get_level_values("ref_motif").str.slice(
        1, 2
    ) + df_motifs.index.get_level_values("alt_motif").str.slice(1, 2)
    df_motifs = df_motifs.sort_values("ord")
    df_for = df_motifs[
        (df_motifs["ord"].str.slice(0, 1) == "C")
        | (df_motifs["ord"].str.slice(0, 1) == "T")
    ].copy()
    df_rev = df_motifs[
        (df_motifs["ord"].str.slice(0, 1) == "A")
        | (df_motifs["ord"].str.slice(0, 1) == "G")
    ].copy()
    df_rev = df_rev.reset_index()
    df_rev["ref_motif"] = df_rev["ref_motif"].apply(revcomp)
    df_rev["alt_motif"] = df_rev["alt_motif"].apply(revcomp)
    df_rev = df_rev.set_index(["ref_motif", "alt_motif"])
    df_err = df_for.rename(
        columns={
            error_rate_column: "error_rate_f",
            snp_count_column: "snp_count_f",
            "coverage": "coverage_f",
        }
    )[["ord", "cycle_skip_status", "error_rate_f", "snp_count_f", "coverage_f"]].join(
        df_rev[[error_rate_column, snp_count_column, "coverage"]].rename(
            columns={
                error_rate_column: "error_rate_r",
                snp_count_column: "snp_count_r",
                "coverage": "coverage_r",
            }
        )
    )
    df_err = (
        df_err.reset_index()
        .sort_values(["ord", "ref_motif"])
        .set_index(["ref_motif", "alt_motif"])
    )

    colors = {"CA": "b", "CG": "orange", "CT": "r", "TA": "gray", "TC": "g", "TG": "m"}
    df_err["color"] = df_err.apply(lambda x: colors.get(x["ord"]), axis=1)
    # plot

    fig, axs = plt.subplots(1, 1, figsize=(24, 6))
    suptitle = plt.title(title, y=1.15, fontsize=32)
    plt.ylabel("Error rate")

    plt.bar(
        np.arange(df_err.shape[0]),
        df_err["error_rate_f"],
        color=df_err["color"],
        width=w,
        zorder=1,
    )
    plt.bar(
        np.arange(df_err.shape[0]) + w,
        df_err["error_rate_r"],
        color="k",
        width=w,
        zorder=1,
    )

    for j, a in enumerate(df_err["ord"].unique()):
        fig.text(
            (j * 0.128) + 0.19,
            0.91,
            f"{a[0]}->{a[1]}",
            ha="center",
            color="w",
            bbox=dict(facecolor=colors[a],),
            fontsize=14,
        )  # fontname='monospace', ha='center', va='center')
    legend = fig.text(
        0.83,
        0.97,
        f"Color - forward strand\nBlack - reverse strand\nmean error = {df_motifs[error_rate_column].mean():.1e}\nmedian error = {df_motifs[error_rate_column].median():.1e}\nSNP number = {df_motifs[snp_count_column].sum():.1e}".replace(
            "e-0", "e-"
        ).replace(
            "e+0", "e+"
        ),
        bbox=dict(facecolor="w", edgecolor="k"),
        fontsize=14,
        ha="center",
    )
    if left_bbox_text is not None:
        fig.text(
            1 - 0.83,
            0.99,
            left_bbox_text,
            bbox=dict(facecolor="w", edgecolor="k"),
            fontsize=14,
            ha="center",
        )

    plt.scatter(
        np.arange(df_err.shape[0]),
        1 / df_err["coverage_f"],
        marker="_",
        color="w",
        linewidth=2,
        zorder=2,
    )
    plt.scatter(
        np.arange(df_err.shape[0]) + w,
        1 / df_err["coverage_r"],
        marker="_",
        color="w",
        linewidth=3,
        zorder=2,
    )
    for j, a in enumerate(df_err["ord"].unique()):
        df_tmp = df_err[df_err["ord"] == a]
        fig.text(
            (j * 0.128 + 0.19),
            -0.05,
            f"{df_tmp['error_rate_f'].mean():.1e}/{df_tmp['error_rate_r'].mean():.1e}".replace(
                "e-0", "e-"
            ),
            color="w",
            ha="center",
            bbox=dict(facecolor=colors[a],),
            fontsize=14,
        )

    xticks = plt.xticks(
        np.arange(df_err.shape[0]) + w / 2,
        [
            ("+ " if c == CYCLE_SKIP else "| ")
            + "$"
            + s[0]
            + r"{\bf "
            + s[1]
            + "}"
            + s[2]
            + "$"
            for s, c in zip(
                df_err.index.get_level_values("ref_motif").values,
                df_err["cycle_skip_status"].values,
            )
        ],
        rotation=90,
        fontsize=12,
        fontname="monospace",
        ha="center",
        va="top",
    )
    plt.yscale(scale)
    plt.xlim(-0.4, 96)

    text1 = fig.text(0.12, 0.065, "Motif:", ha="right")
    text2 = fig.text(0.12, 0.015, "Cycle skip:", ha="right")
    text3 = fig.text(0.12, -0.07, "Mean error -> \n(Forward/Reverse)", ha="center")

    plt.ylim(
        min(df_motifs[error_rate_column].min() * 0.8, 1e-6),
        max(df_motifs[error_rate_column].max() * 1.2, 1e-3),
    )
    if out_filename is None:
        return fig
    else:
        fig.savefig(
            out_filename,
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=[suptitle, legend, text1, text2, text3] + xticks[1],
        )


def call_featuremap_to_dataframe(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    featuremap_to_dataframe(
        featuremap_vcf=args_in.input,
        output_file=args_in.output,
        reference_fasta=args_in.reference_fasta,
        motif_length=args_in.motif_length,
        report_read_orientation=not args_in.report_sense_strand_bases,
        show_progress_bar=args_in.show_progress_bar,
        flow_order=args_in.flow_order,
    )
    sys.stdout.write("DONE\n")


def call_merge_featuremap_dataframes(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    merge_featuremap_dataframes(
        dataframes=args_in.input, outfile=args_in.output, n_jobs=args_in.jobs,
    )
    sys.stdout.write("DONE\n")


def call_collect_coverage_per_motif(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    collect_coverage_per_motif(
        depth_files=args_in.input,
        reference_fasta=args_in.reference_fasta,
        outfile=args_in.output,
        show_stats=args_in.show_stats,
        n_jobs=args_in.jobs,
        size=args_in.motif_length,
        N=args_in.N,
    )


def call_calculate_snp_error_rate(args_in):
    calculate_snp_error_rate(
        single_substitution_featuremap=args_in.featuremap,
        coverage_stats=args_in.coverage_stats,
        depth_data=args_in.depth_data,
        out_path=args_in.output,
        out_basename=args_in.basename,
        reference_fasta=args_in.reference_fasta,
        flow_order=args_in.flow_order,
        featuremap_chrom=args_in.chrom,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_featuremap_to_dataframe = subparsers.add_parser(
        name="to_dataframe", description="""Convert featuremap to pandas dataframe""",
    )
    parser_concat_dataframes = subparsers.add_parser(
        name="concat_dataframes",
        description="""Concat featuremap pandas dataframe created on different intevals""",
    )
    parser_coverage_per_motif = subparsers.add_parser(
        name="collect_coverage_per_motif",
        description="""Collect coverage per motif from a collection of depth files""",
    )
    parser_calculate_snp_error_rate = subparsers.add_parser(
        name="calculate_snp_error_rate",
        description="""Calculate SNP error rate per motif""",
    )

    parser_featuremap_to_dataframe.add_argument(
        "-i", "--input", type=str, required=True, help="input featuremap file",
    )
    parser_featuremap_to_dataframe.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output dataframe will be written, if None a file with the same name as input and 
".parquet" extension will be created""",
    )
    parser_featuremap_to_dataframe.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="""reference fasta, only required for motif annotation
most likely gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta but it must be localized""",
    )
    parser_featuremap_to_dataframe.add_argument(
        "-f",
        "--flow_order",
        type=str,
        required=False,
        default=None,
        help="""flow order - required for cycle skip annotation but not mandatory""",
    )
    parser_featuremap_to_dataframe.add_argument(
        "-m",
        "--motif_length",
        type=int,
        default=4,
        help="motif length to annotate the vcf with",
    )
    parser_featuremap_to_dataframe.add_argument(
        "--report_sense_strand_bases",
        default=False,
        action="store_true",
        help="if True, the ref, alt, and motifs will be reported according to the sense strand and not according to the read orientation",
    )
    parser_featuremap_to_dataframe.add_argument(
        "--show_progress_bar",
        default=False,
        action="store_true",
        help="show progress bar (tqdm)",
    )

    parser_featuremap_to_dataframe.set_defaults(func=call_featuremap_to_dataframe)

    parser_concat_dataframes.add_argument(
        "input", nargs="+", type=str, help="input featuremap files",
    )
    parser_concat_dataframes.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="""Path to which output dataframe will be written, if None a file with the same name as input and 
    ".parquet" extension will be created""",
    )
    parser_concat_dataframes.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel (default 1, -1 for max)",
    )

    parser_concat_dataframes.set_defaults(func=call_merge_featuremap_dataframes)

    parser_coverage_per_motif.add_argument(
        "input", nargs="+", type=str, help="input depth files",
    )
    parser_coverage_per_motif.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="""Path to which output dataframe will be written in hdf format""",
    )
    parser_coverage_per_motif.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="""reference fasta, only required for motif annotation
    most likely gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta but it must be localized""",
    )
    parser_coverage_per_motif.add_argument(
        "-N",
        type=int,
        default=4,
        help="""Process 1 in every N positions - makes the code run faster the larger N is (default 100). In the output
dataframe df, df['coverage'] = df['count'] * N""",
    )
    parser_coverage_per_motif.add_argument(
        "-m",
        "--motif_length",
        type=int,
        default=4,
        help="Maximal motif length to collect coverage for",
    )
    parser_coverage_per_motif.add_argument(
        "--show_stats",
        default=False,
        action="store_true",
        help="Print motif statistics",
    )
    parser_coverage_per_motif.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel (default -1 for max)",
    )
    parser_coverage_per_motif.set_defaults(func=call_collect_coverage_per_motif)

    parser_calculate_snp_error_rate.add_argument(
        "-f",
        "--featuremap",
        type=str,
        required=True,
        help="""Featuremap parquet file""",
    )
    parser_calculate_snp_error_rate.add_argument(
        "--coverage_stats",
        type=str,
        required=True,
        help="""Coverage stats h5 file generated by the coverage_analysis code""",
    )
    parser_calculate_snp_error_rate.add_argument(
        "--depth_data",
        type=str,
        nargs="+",
        required=True,
        help="""Coverage depth bed (gzipped or not) files generated by the coverage_analysis code""",
    )
    parser_calculate_snp_error_rate.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="""Path to which output dataframe will be written (multiple files)""",
    )
    parser_calculate_snp_error_rate.add_argument(
        "--basename",
        type=str,
        default="",
        required=False,
        help="""basename of output files""",
    )
    parser_calculate_snp_error_rate.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="""reference fasta, only required for motif annotation
    most likely gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta but it must be localized""",
    )
    parser_calculate_snp_error_rate.add_argument(
        "--flow_order",
        type=str,
        required=True,
        default=None,
        help="""flow order - required for cycle skip annotation """,
    )
    parser_calculate_snp_error_rate.add_argument(
        "--chrom",
        type=str,
        required=False,
        default=None,
        help="""single chromosome the featuremap was calculated for (leave blank if all chromosomes were included""",
    )
    parser_calculate_snp_error_rate.set_defaults(func=call_calculate_snp_error_rate)

    args = parser.parse_args()
    args.func(args)