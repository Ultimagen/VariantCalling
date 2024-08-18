from __future__ import annotations

import os
from os.path import join as pjoin

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from ugvc import logger
from ugvc.dna.format import CYCLE_SKIP, CYCLE_SKIP_STATUS, DEFAULT_FLOW_ORDER
from ugvc.dna.utils import revcomp
from ugvc.mrd.coverage_utils import collect_coverage_per_motif
from ugvc.utils.misc_utils import read_hdf
from ugbio_core.plotting_utils import set_pyplot_defaults
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe

set_pyplot_defaults()


def calculate_substitution_error_rate(  # pylint: disable=too-many-arguments
    single_substitution_featuremap,
    coverage_stats,
    depth_data,
    out_path: str,
    out_basename: str = "",
    xscore_thresholds: list = None,
    reference_fasta: str = None,
    flow_order: str = DEFAULT_FLOW_ORDER,
    featuremap_chrom: str = None,
    n_jobs=-1,
    coverage_calculation_downsampling_ratio=100,
):
    # init
    if xscore_thresholds is None:
        xscore_thresholds = [0, 3, 5, 10]
    assert len(xscore_thresholds) == 4, f"Length of xscore_thresholds must be 4, got {xscore_thresholds}"

    xscore_thresholds = np.array(xscore_thresholds)  # pylint: disable=redefined-variable-type
    min_xscore = np.min(xscore_thresholds)

    os.makedirs(out_path, exist_ok=True)
    if len(out_basename) > 0 and not out_basename.endswith("."):
        out_basename += "."
    out_coverage_per_motif = pjoin(out_path, f"{out_basename}coverage_per_motif.h5")
    out_snp_rate = pjoin(out_path, f"{out_basename}substitution_error_rate.h5")
    out_snp_rate_plots = {
        th: pjoin(out_path, f"{out_basename}substitution_error_rate_by_motif_thresh{th}.png")
        for th in xscore_thresholds
    }

    # read coverage stats and derive coverage range
    if isinstance(coverage_stats, pd.Series):
        df_coverage_stats = coverage_stats
    else:
        logger.debug(f"Reading input coverage stats from {coverage_stats}")
        df_coverage_stats = read_hdf(coverage_stats, key="histogram").filter(regex="Genome").iloc[:, 0]
    f = interp1d(
        (df_coverage_stats.cumsum() / df_coverage_stats.sum()).values,
        df_coverage_stats.index.values,
        bounds_error=False,
        fill_value=0,
    )
    min_coverage = min(20, np.round(f(0.5)).astype(int))  # the lower between 20 or the median value
    max_coverage = max(np.round(f(0.95)).astype(int), min_coverage + 1)

    x = df_coverage_stats[
        (df_coverage_stats.index.values >= min_coverage) & (df_coverage_stats.index.values <= max_coverage)
    ]
    coverage_correction_factor = (x * x.index.values).sum() / (df_coverage_stats * df_coverage_stats.index).sum()
    logger.debug(
        f"Coverage range {min_coverage}-{max_coverage}x, spanning {coverage_correction_factor:.0%} of the data"
    )
    # generate or read coverage per motif
    if isinstance(depth_data, pd.DataFrame):
        df_coverage = depth_data
    else:
        if reference_fasta is None:
            raise ValueError("Reference fasta must be provided if input is depth files")
        logger.debug("Generating coverage per motif")
        _ = collect_coverage_per_motif(
            depth_data,
            reference_fasta,
            outfile=out_coverage_per_motif,
            show_stats=False,
            n_jobs=n_jobs,
            size=2,
            downsampling_ratio=coverage_calculation_downsampling_ratio,
        )
        df_coverage = pd.read_hdf(out_coverage_per_motif, "motif_2")
    df_coverage.index.name = "ref_motif2"
    # read featuremap
    if isinstance(single_substitution_featuremap, pd.DataFrame):
        df = single_substitution_featuremap
        df = df[
            (df["X_SCORE"] >= min_xscore) & (df["X_READ_COUNT"] >= min_coverage) & (df["X_READ_COUNT"] <= max_coverage)
        ]
    else:
        logger.debug(f"Reading featuremap dataframe from {single_substitution_featuremap}")
        df = pd.read_parquet(
            single_substitution_featuremap,
            filters=[
                ("X_SCORE", ">=", min_xscore),
                ("X_READ_COUNT", ">=", min_coverage),
                ("X_READ_COUNT", "<=", max_coverage),
            ],
        )
        if df.shape[0] == 0:
            raise ValueError(
                f"Length of DataFrame read from {single_substitution_featuremap} is 0, unable to proceed with analysis"
            )
        if not isinstance(df.index, pd.MultiIndex):  # DataFrame was saved as not multi-indexed
            df = df.set_index(["chrom", "pos"])
        if featuremap_chrom is not None:
            logger.debug(f"using only data in {featuremap_chrom}")
            df = df.loc[featuremap_chrom]
    # calculate read filtration ratio in featuremap
    read_filter_correction_factor = (df["X_FILTERED_COUNT"] + 1).sum() / df["X_READ_COUNT"].sum()
    # process 2nd order motifs
    logger.debug("Processing motifs")
    left = (
        df["left_motif"]
        .astype(str)
        .str.slice(
            2,
        )
    )
    right = df["right_motif"].astype(str).str.slice(0, -2)

    df["ref_motif2"] = left + df["ref"].astype(str) + right
    df["alt_motif2"] = left + df["alt"].astype(str) + right
    logger.debug("Grouping by 2nd order motif")
    df_motifs_2 = df.groupby(["ref_motif2", "alt_motif2"]).agg(
        {
            **{x: "first" for x in ("ref", "alt", "ref_motif", "alt_motif")},
            **{
                "X_SCORE": [
                    lambda a: np.sum(a >= xscore_thresholds[0]).astype(int),
                    lambda a: np.sum(a >= xscore_thresholds[1]).astype(int),
                    lambda a: np.sum(a >= xscore_thresholds[2]).astype(int),
                    lambda a: np.sum(a >= xscore_thresholds[3]).astype(int),
                ]
            },
        }
    )
    df_motifs_2 = df_motifs_2.dropna(how="all")

    df_motifs_2.columns = ["ref", "alt", "ref_motif", "alt_motif"] + [f"snp_count_bq{th}" for th in xscore_thresholds]
    logger.debug("Annotating cycle skip")
    df_motifs_2 = (
        df_motifs_2.reset_index()
        .set_index(["ref_motif", "alt_motif"])
        .join(get_cycle_skip_dataframe(flow_order=flow_order))
    )

    df_motifs_2 = df_motifs_2.reset_index().set_index("ref_motif2").join(df_coverage[["coverage"]])
    df_motifs_2.loc[:, "coverage"] = (
        df_motifs_2["coverage"] * read_filter_correction_factor * coverage_correction_factor
    )

    # process 1st order motifs
    logger.debug("Creating 1st order motif data")
    df_motifs_1 = (
        df_motifs_2.groupby(["ref_motif", "alt_motif"])
        .agg(
            {
                **{"ref": "first", "alt": "first", "cycle_skip_status": "first"},
                **{c: "sum" for c in df_motifs_2.columns if "snp_count" in c or "coverage" in c},
            }
        )
        .dropna(how="all")
    )
    # process 0 order motifs
    logger.debug("Creating 0 order motif data")
    df_motifs_0 = (
        df_motifs_1.groupby(["ref", "alt"])
        .agg({c: "sum" for c in df_motifs_1.columns if "snp_count" in c or "coverage" in c})
        .dropna(how="all")
    )
    # process average error rate regardless of motifs
    df_sum = (
        df_motifs_1.assign(coverage_csk=df_motifs_1["coverage"].where(df_motifs_1[CYCLE_SKIP_STATUS] == CYCLE_SKIP))
        .groupby("ref_motif")
        .agg(
            {
                **{"coverage": "first", "coverage_csk": "first"},
                **{f"snp_count_bq{x}": "sum" for x in xscore_thresholds},
            }
        )
        .sum()
    )
    df_avg = df_sum.filter(regex="count").to_frame().T
    for x in xscore_thresholds:
        coverage = df_sum["coverage"] if x < 6 else df_sum["coverage_csk"]
        df_avg.loc[:, f"error_rate_bq{x}"] = df_avg[f"snp_count_bq{x}"] / coverage
    df_avg = df_avg.filter(regex="error_rate").loc[0]

    logger.debug("Setting non-cycle skip motifs at X_SCORE>=6 to NaN")
    for th in xscore_thresholds:
        if th >= 6:
            df_motifs_2.loc[:, f"snp_count_bq{th}"] = df_motifs_2[f"snp_count_bq{th}"].where(
                df_motifs_2[CYCLE_SKIP_STATUS] == CYCLE_SKIP
            )
            df_motifs_1.loc[:, f"snp_count_bq{th}"] = df_motifs_1[f"snp_count_bq{th}"].where(
                df_motifs_1[CYCLE_SKIP_STATUS] == CYCLE_SKIP
            )

    logger.debug("Assigning error rates")
    for df_tmp in (df_motifs_0, df_motifs_1, df_motifs_2):
        for th in xscore_thresholds:
            df_tmp.loc[:, f"error_rate_bq{th}"] = df_tmp[f"snp_count_bq{th}"] / (df_tmp["coverage"])

    # save
    logger.debug(f"Saving to {out_snp_rate}")
    df_motifs_2 = df_motifs_2.reset_index().astype(
        {
            c: "category"
            for c in (
                "ref",
                "alt",
                "ref_motif",
                "alt_motif",
                "ref_motif2",
                "alt_motif2",
            )
        }
    )
    df_motifs_1 = df_motifs_1.reset_index().astype(
        {
            c: "category"
            for c in (
                "ref",
                "alt",
                "ref_motif",
                "alt_motif",
            )
        }
    )
    df_motifs_0 = df_motifs_0.reset_index().astype({c: "category" for c in ("ref", "alt")})
    df_avg.to_hdf(out_snp_rate, key="average", mode="w", format="table")
    df_motifs_0.to_hdf(out_snp_rate, key="motif_0", mode="a", format="table")
    df_motifs_1.to_hdf(out_snp_rate, key="motif_1", mode="a", format="table")
    df_motifs_2.to_hdf(out_snp_rate, key="motif_2", mode="a", format="table")

    # generate plots
    # by_mut_type_and_source
    substitution_error_rate_by_mut_type_and_source = plot_substitution_error_rate_by_mut_type_and_source(
        df_motifs_1,
        out_filename=pjoin(out_path, f"{out_basename}substitution_error_rate_by_mut_type_and_source.png"),
        title=out_basename,
    )
    substitution_error_rate_asymmetry = plot_substitution_error_rate_asymmetry(
        df_motifs_1,
        out_filename=pjoin(out_path, f"{out_basename}substitution_error_rate_asymmetry.png"),
        title=out_basename,
    )

    # error rate by motif
    for th in xscore_thresholds:
        logger.debug(f"Generating plot for X_SCORE>={th}")
        error_rate_column = f"error_rate_bq{th}"
        snp_count_column = f"snp_count_bq{th}"
        plot_substitution_error_rate_by_motif(
            df_motifs_1,
            out_filename=out_snp_rate_plots[th],
            title=f"{out_basename.replace('.', ' ')}\nLog-likelihood threshold = {th}",
            left_bbox_text=f"Coverage range {min_coverage}-{max_coverage}x\n"
            f"spanning {coverage_correction_factor:.0%} of the data",
            error_rate_column=error_rate_column,
            snp_count_column=snp_count_column,
        )

    substitution_error_rate_by_mut_type_and_source.to_hdf(
        out_snp_rate, key="substitution_error_rate_by_mut_type_and_source", mode="a", format="table"
    )
    substitution_error_rate_asymmetry.to_hdf(
        out_snp_rate, key="substitution_error_rate_asymmetry", mode="a", format="table"
    )


def create_matched_forward_and_reverse_strand_errors_dataframe(df_motifs):
    df_motifs = df_motifs.astype({"ref_motif": str, "alt_motif": str})
    df_motifs.loc[:, "mut_type"] = (
        df_motifs["ref_motif"].str.slice(1, 2) + "->" + df_motifs["alt_motif"].str.slice(1, 2)
    )
    df_for = df_motifs[(df_motifs["ref"] == "C") | (df_motifs["ref"] == "T")].copy()
    df_rev = df_motifs[(df_motifs["ref"] == "A") | (df_motifs["ref"] == "G")].copy()
    df_rev.loc[:, "ref_motif"] = df_rev["ref_motif"].apply(revcomp)
    df_rev.loc[:, "alt_motif"] = df_rev["alt_motif"].apply(revcomp)
    df_rev = df_rev.set_index(["ref_motif", "alt_motif"])
    df_for = df_for.set_index(["ref_motif", "alt_motif"])
    df_err = df_for.filter(regex="mut_type|error_rate").join(
        df_rev.filter(regex="error_rate"), lsuffix="_f", rsuffix="_r"
    )
    return df_err


def plot_substitution_error_rate_by_motif(
    df_motifs: pd.DataFrame,
    out_filename: str = None,
    title: str = "",
    left_bbox_text: str = None,
    error_rate_column: str = "error_rate",
    snp_count_column: str = "snp_count",
    backend: str = None,
):
    # init
    assert snp_count_column in df_motifs
    assert error_rate_column in df_motifs
    if "ref_motif" not in df_motifs.index.names:
        assert "ref_motif" in df_motifs
        df_motifs = df_motifs.set_index(["ref_motif", "alt_motif"])

    if backend:
        matplotlib.use(backend)
    elif out_filename is None:
        matplotlib.use("Qt5Agg")
    else:
        matplotlib.use("Agg")  # non interactive
    w = 0.3
    scale = "log"

    # create matched forward-reverse dataframe
    df_motifs["ord"] = df_motifs.index.get_level_values("ref_motif").str.slice(1, 2) + df_motifs.index.get_level_values(
        "alt_motif"
    ).str.slice(1, 2)
    df_motifs = df_motifs.sort_values("ord")
    df_for = df_motifs[(df_motifs["ord"].str.slice(0, 1) == "C") | (df_motifs["ord"].str.slice(0, 1) == "T")].copy()
    df_rev = df_motifs[(df_motifs["ord"].str.slice(0, 1) == "A") | (df_motifs["ord"].str.slice(0, 1) == "G")].copy()
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
        .sort_values(["ord", "cycle_skip_status", "ref_motif"])
        .set_index(["ref_motif", "alt_motif"])
    )
    mean_err = df_motifs[error_rate_column].mean()
    mean_err_cskp = df_motifs.query("cycle_skip_status == 'cycle-skip'")[error_rate_column].mean()
    median_err = df_motifs[error_rate_column].median()
    median_err_cskp = df_motifs.query("cycle_skip_status == 'cycle-skip'")[error_rate_column].median()

    colors = {"CA": "b", "CG": "orange", "CT": "r", "TA": "gray", "TC": "g", "TG": "m"}
    df_err["color"] = df_err.apply(lambda x: colors.get(x["ord"]), axis=1)
    # plot

    fig, _ = plt.subplots(1, 1, figsize=(24, 6))
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
            bbox=dict(
                facecolor=colors[a],
            ),
            fontsize=14,
        )  # fontname='monospace', ha='center', va='center')
    legend = fig.text(
        0.72,
        0.97,
        f"Color - forward strand\n"
        f"Black - reverse strand\n"
        f"mean error (cskp / all) = {mean_err_cskp:.1e} / {mean_err:.1e}\n"
        f"median error (cskp / all) = {median_err_cskp:.1e} / {median_err:.1e}\n"
        f"Number of substitutions = {df_motifs[snp_count_column].sum():.1e}".replace("e-0", "e-").replace("e+0", "e+"),
        bbox=dict(facecolor="w", edgecolor="k"),
        fontsize=14,
        ha="left",
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
            f"{df_tmp['error_rate_f'].mean():.1e}/{df_tmp['error_rate_r'].mean():.1e}".replace("e-0", "e-"),
            color="w",
            ha="center",
            bbox=dict(
                facecolor=colors[a],
            ),
            fontsize=14,
        )

    _, xtickfonts = plt.xticks(
        np.arange(df_err.shape[0]) + w / 2,
        [
            ("+ " if c == CYCLE_SKIP else "  ") + "$" + s[0] + r"{\bf " + s[1] + "}" + s[2] + "$"
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
    for xt, c in zip(xtickfonts, df_err["cycle_skip_status"].values):
        if c == CYCLE_SKIP:
            xt.set_color("xkcd:brick red")
        else:
            xt.set_color("k")
    plt.yscale(scale)
    plt.xlim(-0.4, 96)

    text1 = fig.text(0.12, 0.065, "Motif:", ha="right")
    text2 = fig.text(0.12, 0.015, "Cycle skip:", ha="right")
    text3 = fig.text(0.12, -0.07, "Mean error -> \n(Forward/Reverse)", ha="center")

    plt.ylim(
        min(df_motifs[error_rate_column].min() * 0.8, 1e-6),
        max(df_motifs[error_rate_column].max() * 1.2, 1e-3),
    )

    if out_filename is not None:
        fig.savefig(
            out_filename,
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=[suptitle, legend, text1, text2, text3],
        )
    return fig


def plot_substitution_error_rate_by_mut_type_and_source(
    df_motifs: pd.DataFrame,
    out_filename: str = None,
    title: str = "",
):
    # merge forward and reverse
    df_err = create_matched_forward_and_reverse_strand_errors_dataframe(df_motifs)
    # add average error columns
    for c in df_err.filter(regex="_f").columns:
        df_err.loc[:, c[:-2]] = df_err.filter(regex=c[:-1]).mean(axis=1)

    # calculate errors by contribution
    cskp_col = "error_rate_bq10"
    other_cols = df_err.columns[-4:-1]
    df_err_agg = df_err[["mut_type"] + list(other_cols)][df_err[cskp_col].isnull()].groupby("mut_type").mean()
    df_err_agg = df_err_agg.rename(
        columns={c: c.replace("error_rate_bq", "seq_error_rate_thresh") for c in df_err_agg.columns}
    )
    df_err_agg = df_err_agg.join(
        df_err.groupby("mut_type").agg({cskp_col: "mean"}).rename(columns={cskp_col: "non_sequencing_errors"})
    )
    for c in df_err_agg.filter(regex="seq_err").columns:
        df_err_agg.loc[:, c] = df_err_agg[c] - df_err_agg["non_sequencing_errors"]

    # plot
    colors = {
        "C->A": "b",
        "C->G": "orange",
        "C->T": "r",
        "T->A": "gray",
        "T->C": "g",
        "T->G": "m",
    }
    fig = plt.figure(figsize=(16, 6))
    title = plt.title(title, fontsize=32)
    # boxes
    plt.boxplot(
        df_err_agg,
        positions=range(df_err_agg.shape[1]),
        labels=[
            c.replace("seq_error_rate_thresh", "Sequencing errors\nLog-likelihood thresh ")
            .replace("_", " ")
            .capitalize()
            for c in df_err_agg
        ],
        showfliers=False,
        **{x: dict(alpha=0.5) for x in ("boxprops", "medianprops", "meanprops", "capprops", "whiskerprops")},
    )
    # lines
    for mt, row in df_err_agg.iterrows():
        color = colors[mt]
        plt.plot(
            row.values,
            c=color,
            linestyle="none",
            marker="o",
            label=f"{mt[0]}:{revcomp(mt[0])}→{mt[-1]}:{revcomp(mt[-1])}",
        )
        plt.plot(row.values, c=color, linestyle="-", alpha=0.1)
        x = row["non_sequencing_errors"]
        plt.text(df_err_agg.shape[1] - 1 + 0.03, x, f"{x:.1e}".replace("-0", "-"), ha="left", va="center", color=color)
    legend = plt.legend(title="Mutation type", fontsize=18, title_fontsize=22)
    plt.yscale("log")
    ylim = plt.gca().get_ylim()
    plt.plot([2.5, 2.5], ylim, "-k")
    plt.ylim(ylim)
    ylabel = plt.ylabel("Error rate")

    if out_filename is None:
        return df_err_agg, fig

    fig.savefig(
        out_filename,
        dpi=150,
        bbox_inches="tight",
        bbox_extra_artists=[title, ylabel, legend],
    )

    return df_err_agg


def plot_substitution_error_rate_asymmetry(
    df_motifs: pd.DataFrame,
    out_filename: str = None,
    title: str = "",
):
    colors = {
        "C->A": "b",
        "C->G": "orange",
        "C->T": "r",
        "T->A": "gray",
        "T->C": "g",
        "T->G": "m",
    }

    df_err = create_matched_forward_and_reverse_strand_errors_dataframe(df_motifs)
    df_err.loc[:, "cskp_err_rate_asym_log2"] = np.log2(df_err["error_rate_bq10_f"] / df_err["error_rate_bq10_r"])
    df_err = (
        df_err.dropna(subset=["cskp_err_rate_asym_log2"])
        .set_index("mut_type")[["cskp_err_rate_asym_log2"]]
        .sort_index()
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 8), sharex=True)
    title = plt.title(title, fontsize=32)
    yticks = []
    color_list = []
    for j, (mt, group) in enumerate(df_err.groupby("mut_type")):
        plt.boxplot(
            group,
            positions=[j],
            vert=False,
            showfliers=False,
        )
        label = f"{mt[0]}→{mt[-1]} / {revcomp(mt[0])}→{revcomp(mt[-1])}"
        color = colors[mt]
        color_list.append(color)
        mean_asym = np.log2((2**group).mean().values[0])
        plt.scatter(
            mean_asym,
            j + 0.17,
            s=50,
            marker="^",
            c=color,
            label=label,
        )
        plt.text(
            mean_asym,
            j + 0.25,
            f"x{2 ** mean_asym:.1f}",
            color=color,
            ha="center",
        )
        yticks.append(label)
    xlim = ax.get_xlim()
    ax.set_xlim(min(xlim[0], -2), max(xlim[1], 2))
    _, ticklabels = plt.yticks(range(len(yticks)), yticks)
    for xt, c in zip(ticklabels, color_list):
        xt.set_color(c)
    plt.xlabel("$log_2$ error asymmetry\nCycle-skip motifs")

    if out_filename is None:
        return df_err, fig

    fig.savefig(
        out_filename,
        dpi=150,
        bbox_inches="tight",
        bbox_extra_artists=[title],
    )

    return df_err
