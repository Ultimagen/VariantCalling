from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ugvc.dna.format import CYCLE_SKIP, CYCLE_SKIP_STATUS, DEFAULT_FLOW_ORDER
from ugvc.dna.utils import revcomp
from ugvc.utils.cloud_sync import cloud_sync
from ugvc.utils.consts import FileExtension
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe

POSITIONAL_MAX_POSITIONS_PLOT = 300
POSITIONAL_MIN_POSITIONS_PLOT = 0
POSITIONAL_BINS_BIAS = 0.5
FONT_SIZE = 28
REFERENCE = "ref"
POSITIONAL_PLOT_STEP_SIZE = 1
POSITIONAL_X_EDIST_THRESHOLD = 4
POSITIONAL_X_SCORE_THRESHOLD = 10
POSITIONAL_DEFAULT_FILE_PREFIX = "mutation_position_histogram_per_motif"
POSITIONAL_DEFAULT_ERROR_PER_PREFIX = "error_per_position_per_motif"


class FeatureMapColumnName(Enum):
    """Column names enum of SNP feature map dataframe (read from parquet file)"""

    CIGAR = "X_CIGAR"
    EDIT_DISTANCE = "X_EDIST"
    SCORE = "X_SCORE"
    INDEX = "X_INDEX"
    LENGTH = "X_LENGTH"
    REFERENCE_MOTIF = "ref_motif"
    ALTERNATE_MOTIF = "alt_motif"
    ALTERNATIVE = "alt"
    CYCLE_SKIP_STATUS = "cycle_skip_status"


class ErrorRateColumnName(Enum):
    """Column names enum of error rate dataframe"""

    NAME = "name"
    REF = "ref"
    ALT = "alt"
    INDEX = "X_INDEX"
    COUNT_FORWARD = "count_forward"
    COUNT_REVERSE = "count_reverse"
    LENGTH_CDF = "length_cdf"
    COVERAGE = "coverage"


def preprocess_filename(file_name):
    return os.path.basename(file_name).split(".")[0]


def load_substitutions(single_substitutions_file_name: str) -> pd.DataFrame:
    """Load SNP substitutions file, return it's DataFrame"""
    out = pd.read_parquet(
        cloud_sync(single_substitutions_file_name),
    )
    if not isinstance(out.index, pd.MultiIndex):  # DataFrame was saved as not multi-indexed
        out = out.set_index(["chrom", "pos"])
    out = out.assign(
        **{
            ErrorRateColumnName.REF.value: out[FeatureMapColumnName.REFERENCE_MOTIF.value].str.slice(1, 2),
            ErrorRateColumnName.ALT.value: out[FeatureMapColumnName.ALTERNATE_MOTIF.value].str.slice(1, 2),
        }
    )
    return out


def filter_substitutions(
    substitutions_df: pd.DataFrame,
    allow_softlclip: bool = False,
    edit_distance_threshold: int = POSITIONAL_X_EDIST_THRESHOLD,
    xscore_threshold: int = POSITIONAL_X_SCORE_THRESHOLD,
):
    """Filter out the substitutions that don't appear above thresholds of edit distance and x score"""
    if allow_softlclip:
        threshold_filtered_df = substitutions_df[
            (substitutions_df[FeatureMapColumnName.EDIT_DISTANCE.value] <= edit_distance_threshold)
            & (substitutions_df[CYCLE_SKIP_STATUS] == CYCLE_SKIP)
            & (substitutions_df[FeatureMapColumnName.SCORE.value] >= xscore_threshold)
        ]
    else:
        threshold_filtered_df = substitutions_df[
            (substitutions_df[FeatureMapColumnName.EDIT_DISTANCE.value] <= edit_distance_threshold)
            & (substitutions_df[FeatureMapColumnName.SCORE.value] >= xscore_threshold)
            & (substitutions_df[CYCLE_SKIP_STATUS] == CYCLE_SKIP)
            & (~substitutions_df[FeatureMapColumnName.CIGAR.value].str.contains("S"))
        ]

    return threshold_filtered_df


def load_coverage_per_motif(coverage_file_name, motif_name):
    out = pd.read_hdf(
        cloud_sync(coverage_file_name),
        motif_name,
    )
    out.index.name = FeatureMapColumnName.REFERENCE_MOTIF.value
    return out


def get_cycle_skip_motifs(flow_order: str = DEFAULT_FLOW_ORDER):
    """Get cycle skip motifs"""
    cycle_skip_df = get_cycle_skip_dataframe(flow_order)
    ref_cskp_motifs = cycle_skip_df[cycle_skip_df[CYCLE_SKIP_STATUS] == CYCLE_SKIP]
    ref_cskp_motifs = ref_cskp_motifs.assign(
        ref=[x[1] for x in ref_cskp_motifs.index.get_level_values("ref_motif")],
        alt=[x[1] for x in ref_cskp_motifs.index.get_level_values("alt_motif")],
    )

    return ref_cskp_motifs


@lru_cache()
def _get_bins(
    max_positions_plot: int = POSITIONAL_MAX_POSITIONS_PLOT,
    position_plot_step_size: int = POSITIONAL_PLOT_STEP_SIZE,
):
    return (
        np.arange(POSITIONAL_MIN_POSITIONS_PLOT, max_positions_plot + 1, position_plot_step_size) - POSITIONAL_BINS_BIAS
    )


def calculate_positional_substitution_error_rate(
    run_name: str,
    single_substitutions_df,
    coverage_df,
    output_folder: str,
    error_per_position_per_motif: str,
    max_positions_plot: int = POSITIONAL_MAX_POSITIONS_PLOT,
    position_plot_step_size: int = POSITIONAL_PLOT_STEP_SIZE,
):
    error_rate_results = []

    bins = _get_bins(max_positions_plot, position_plot_step_size)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    h_length, _ = np.histogram(single_substitutions_df[FeatureMapColumnName.LENGTH.value], bins=bins)
    h_length_cumsum = pd.DataFrame(dict(length_CDF=np.cumsum(h_length[::-1])[::-1] / sum(h_length)))
    h_length_cumsum.index.name = ErrorRateColumnName.INDEX.value

    for ind, group in single_substitutions_df.groupby([ErrorRateColumnName.REF.value, ErrorRateColumnName.ALT.value]):
        df_index_hist = pd.DataFrame(
            data={
                FeatureMapColumnName.INDEX.value: bin_centers,
                ErrorRateColumnName.COUNT_FORWARD.value: np.histogram(
                    group[FeatureMapColumnName.INDEX.value],
                    bins=bins,
                )[0],
            }
        )
        df_index_hist = df_index_hist.assign(
            **{
                ErrorRateColumnName.LENGTH_CDF.value: df_index_hist[ErrorRateColumnName.COUNT_FORWARD.value].cumsum(),
                ErrorRateColumnName.COVERAGE.value: coverage_df.loc[ind, ErrorRateColumnName.COVERAGE.value],
                ErrorRateColumnName.REF.value: ind[0],
                ErrorRateColumnName.ALT.value: ind[1],
                ErrorRateColumnName.NAME.value: run_name,
            }
        ).set_index(
            [
                ErrorRateColumnName.NAME.value,
                ErrorRateColumnName.REF.value,
                ErrorRateColumnName.ALT.value,
                FeatureMapColumnName.INDEX.value,
            ]
        )
        error_rate_results.append(df_index_hist)

    error_rate_results_df = pd.concat(error_rate_results)
    error_rate_results_df_for = error_rate_results_df.loc[([run_name, "C", "T"], slice(None)), :]
    error_rate_results_df_rev = (
        error_rate_results_df.loc[([run_name, "G", "A"], slice(None)), :]
        .reset_index()
        .rename(columns={ErrorRateColumnName.COUNT_FORWARD.value: ErrorRateColumnName.COUNT_REVERSE.value})
    )
    error_rate_results_df_rev.loc[:, ErrorRateColumnName.REF.value] = error_rate_results_df_rev[
        ErrorRateColumnName.REF.value
    ].apply(revcomp)

    error_rate_results_df_rev.loc[:, ErrorRateColumnName.ALT.value] = error_rate_results_df_rev[
        ErrorRateColumnName.ALT.value
    ].apply(revcomp)

    error_rate_results_df_rev = error_rate_results_df_rev.set_index(
        [
            ErrorRateColumnName.NAME.value,
            ErrorRateColumnName.REF.value,
            ErrorRateColumnName.ALT.value,
            FeatureMapColumnName.INDEX.value,
        ]
    )

    error_rate_results_df = (
        error_rate_results_df_for.join(
            error_rate_results_df_rev[ErrorRateColumnName.COUNT_REVERSE.value], rsuffix="_REV"
        )
        .reset_index(level=FeatureMapColumnName.INDEX.value)
        .astype({FeatureMapColumnName.INDEX.value: int})
    )

    # save the results to parquet file
    error_rate_results_df.reset_index().to_parquet(
        os.path.join(
            output_folder,
            f"{run_name}.{error_per_position_per_motif}{FileExtension.PARQUET.value}",
        )
    )

    return error_rate_results_df


def _create_plot(run_name: str) -> tuple:
    fig, axs = plt.subplots(2, 3, figsize=(20, 6), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    artists = [fig.suptitle(f"{run_name}", y=1.05)]
    return fig, axs, artists


def calc_positional_error_rate_profile(  # pylint: disable=too-many-arguments
    single_substitutions_file_name: str,
    coverage_per_motif_file_name: str,
    output_folder: str,
    output_file_prefix: str = POSITIONAL_DEFAULT_FILE_PREFIX,
    error_per_pos_file_prefix: str = POSITIONAL_DEFAULT_ERROR_PER_PREFIX,
    allow_softlclip: bool = False,
    flow_order: str = DEFAULT_FLOW_ORDER,
    position_plot_step_size: int = POSITIONAL_PLOT_STEP_SIZE,
    max_positions_plot: int = POSITIONAL_MAX_POSITIONS_PLOT,
    edist_threshold: int = POSITIONAL_X_EDIST_THRESHOLD,
    xscore_threshold: int = POSITIONAL_X_SCORE_THRESHOLD,
):
    """
    Actual process of calculating error rate profile by base position within entire sequence
    :param single_substitutions_file_name: single substitutions file name to use, this file is a parquet file
    :param coverage_per_motif_file_name: coverage per motif file name to use, this file is a h5 file
    :param output_folder: output folder to save graph result files
    :param output_file_prefix: name of output result file
    :param error_per_pos_file_prefix: prefix of error per position result file
    :param allow_softlclip: allow include softclip within calculations
    :param flow_order: flow order of nucleotides
    :param position_plot_step_size: step size for plotting graph
    :param max_positions_plot: max positions for plotting graph
    :param edist_threshold: edit distance threshold to use for calculations
    :param xscore_threshold: s-score threshold to use for calculations
    """
    # preprocess
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    run_name = preprocess_filename(single_substitutions_file_name)

    # read feature maps
    single_substitutions_df = load_substitutions(single_substitutions_file_name)
    single_substitutions_df = filter_substitutions(
        single_substitutions_df, allow_softlclip, edist_threshold, xscore_threshold
    )
    coverage_df = load_coverage_per_motif(coverage_per_motif_file_name, "motif_1")
    ref_cskp_motifs = get_cycle_skip_motifs(flow_order)

    # Analysis
    # sum the coverage per (ref, alt) for cycle skip motifs (not the same number of cskp motifs for every type)
    coverage_df = ref_cskp_motifs.join(coverage_df).groupby(["ref", "alt"]).agg({"coverage": "sum"})

    # process
    error_rate_results = calculate_positional_substitution_error_rate(
        run_name,
        single_substitutions_df,
        coverage_df,
        output_folder,
        error_per_pos_file_prefix,
        max_positions_plot,
        position_plot_step_size,
    )

    # Create graphs
    plot_positional_substitution_error_rate(
        error_rate_results,
        run_name,
        output_folder,
        output_file_prefix,
        max_positions_plot,
        position_plot_step_size,
    )


def plot_positional_substitution_error_rate(
    error_rate_results,
    run_name,
    output_folder,
    output_file_prefix: str,
    max_positions_plot: int = POSITIONAL_MAX_POSITIONS_PLOT,
    position_plot_step_size: int = POSITIONAL_PLOT_STEP_SIZE,
):
    """Create output graphs of positional SNP error rate for each variant"""
    bins = _get_bins(max_positions_plot, position_plot_step_size)

    tight_xlim = error_rate_results[error_rate_results.filter(regex="count").max(axis=1) > 10][
        FeatureMapColumnName.INDEX.value
    ].max()

    # Create basic plot
    fig, axs, artists = _create_plot(run_name)

    for ax_flatten, (error_rate_run_name, ref, alt) in zip(axs.flatten(), error_rate_results.index.unique()):
        error_rate_plot = error_rate_results.loc[(error_rate_run_name, ref, alt), :]
        plt.sca(ax_flatten)
        plt.title(f"{ref}->{alt}", fontsize=FONT_SIZE)

        for column, color, label in (
            (ErrorRateColumnName.COUNT_FORWARD.value, "b", f"{ref}->{alt}"),
            (
                ErrorRateColumnName.COUNT_REVERSE.value,
                "r",
                f"{revcomp(ref)}->{revcomp(alt)}",
            ),
        ):
            norm = error_rate_plot[ErrorRateColumnName.COVERAGE.value] * (
                error_rate_plot[ErrorRateColumnName.LENGTH_CDF.value]
                / error_rate_plot[ErrorRateColumnName.LENGTH_CDF.value].sum()
            )
            plt.plot(
                error_rate_plot[ErrorRateColumnName.INDEX.value],
                error_rate_plot[column].mask((error_rate_plot[column] < 10)) / norm,
                label=f"{label} = {error_rate_plot[column].sum() / norm.sum():.1E}".replace("E", r" $\cdot$ ").replace(
                    "-0", "$10^{-"
                )
                + "}$",
                c=color,
                linewidth=2,
            )
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax_flatten.yaxis.set_major_formatter(formatter)
        plt.ylim(0, ax_flatten.get_ylim()[1])
        plt.legend(fontsize="medium", title="Mean error rate")

        if ref == "T":
            artists.append(plt.xlabel("Mutation position"))
        if alt == "A":
            artists.append(plt.ylabel("Substitution error", fontsize=16))

    plt.xlim(bins[0], bins[-1])
    fig.savefig(
        os.path.join(
            output_folder,
            f"{run_name}.{output_file_prefix}{FileExtension.PNG.value}",
        ),
        bbox_extra_artists=artists,
        bbox_inches="tight",
    )
    plt.xlim([0, tight_xlim])
    plt.legend(loc="upper right", fontsize="medium")
    fig.savefig(
        os.path.join(
            output_folder,
            f"{run_name}.{output_file_prefix}.tight_xlim{FileExtension.PNG.value}",
        ),
        bbox_extra_artists=artists,
        bbox_inches="tight",
    )
