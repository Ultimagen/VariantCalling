import argparse
import os
import os.path
from enum import Enum
from functools import lru_cache
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


from ugvc.dna.format import CYCLE_SKIP, CYCLE_SKIP_STATUS, DEFAULT_FLOW_ORDER
from ugvc.dna.utils import revcomp
from ugvc.utils.cloud_sync import cloud_sync
from ugvc.utils.consts import COVERAGE, FileExtension
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe

MAX_POSITIONS_PLOT = 300
MIN_POSITIONS_PLOT = 0
BINS_BIAS = 0.5
FONT_SIZE = 28
REFERENCE = "ref"

# Default Values for input arguments
POSITION_PLOT_STEP_SIZE = 1
TH_X_EDIST = 4
TH_X_SCORE = 10

DEFAULT_FILE_PREFIX = "mutation_position_histogram_per_motif"
DEFAULT_ERROR_PER_POSITION_PREFIX = "error_per_position_per_motif"


class FeatureMapColumnName(Enum):
    """ Column names enum of SNP feature map dataframe (read from parquet file)"""
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
    """ Column names enum of error rate dataframe """
    NAME = "name"
    REF = "ref"
    ALT = "alt"
    INDEX = "X_INDEX"
    COUNT_FORWARD = "count_forward"
    COUNT_REVERSE = "count_reverse"
    LENGTH_CDF = "length_cdf"
    COVERAGE = "coverage"


#  pre-process functions #


def preprocess_filename(file_name):
    return os.path.basename(file_name).split(".")[0].replace("-", "_")


def load_substitutions(single_substitutions_file_name: str) -> pd.DataFrame:
    """Load SNP substitutions file, return it's DataFrame"""
    return pd.read_parquet(
        cloud_sync(single_substitutions_file_name),
        columns=[column.value for column in FeatureMapColumnName],
    )


def filter_substitutions_threshold(
    substitutions_df: pd.DataFrame,
    allow_softlclip: bool = False,
    edit_distance_threshold: int = TH_X_EDIST,
    xscore_threshold: int = TH_X_SCORE,
):
    """ Filter out the substitutions that don't appear above thresholds of edit distance and x score """
    if allow_softlclip:
        threshold_filtered_df = substitutions_df[
            (
                substitutions_df[FeatureMapColumnName.EDIT_DISTANCE.value]
                <= edit_distance_threshold
            )
            & (substitutions_df[FeatureMapColumnName.SCORE.value] >= xscore_threshold)
        ]
    else:
        threshold_filtered_df = substitutions_df[
            (
                substitutions_df[FeatureMapColumnName.EDIT_DISTANCE.value]
                <= edit_distance_threshold
            )
            & (substitutions_df[FeatureMapColumnName.SCORE.value] >= xscore_threshold)
            & (~substitutions_df[FeatureMapColumnName.CIGAR.value].str.contains("S"))
        ]

    return threshold_filtered_df


def load_coverage_per_motif(coverage_file_name, motif_name):
    return pd.read_hdf(
        cloud_sync(coverage_file_name),
        motif_name,
    )


#  analysis functions #


def get_cycle_skip_motifs(flow_order: str = DEFAULT_FLOW_ORDER):
    """ Get cycle skip motifs """
    cycle_skip_df = get_cycle_skip_dataframe(flow_order)

    ref_cskp_motifs = cycle_skip_df[cycle_skip_df[CYCLE_SKIP_STATUS] == CYCLE_SKIP]
    ref_cskp_motifs = (
        ref_cskp_motifs.assign(
            ref=[
                x[1]
                for x in ref_cskp_motifs.index.get_level_values(
                    FeatureMapColumnName.REFERENCE_MOTIF.value
                )
            ],
            alt=[
                x[1]
                for x in ref_cskp_motifs.index.get_level_values(
                    FeatureMapColumnName.ALTERNATE_MOTIF.value
                )
            ],
        )
        .reset_index()
        .set_index([REFERENCE, FeatureMapColumnName.ALTERNATIVE.value])
        .loc[["C", "T"]]
        .sort_index()
    )

    return ref_cskp_motifs


#  process functions #


@lru_cache()
def _get_bins(
    max_positions_plot: int = MAX_POSITIONS_PLOT,
    position_plot_step_size: int = POSITION_PLOT_STEP_SIZE,
):
    return (
        np.arange(MIN_POSITIONS_PLOT, max_positions_plot + 1, position_plot_step_size)
        - BINS_BIAS
    )


def calc_error_rate(  # pylint: disable=too-many-arguments
    run_name: str,
    ref_cskp_motifs,
    single_substitutions_df,
    coverage_df,
    output_folder: str,
    error_per_position_per_motif: str,
    max_positions_plot: int = MAX_POSITIONS_PLOT,
    position_plot_step_size: int = POSITION_PLOT_STEP_SIZE,
):
    error_rate_results = []

    bins = _get_bins(max_positions_plot, position_plot_step_size)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    h_length, _ = np.histogram(
        single_substitutions_df[FeatureMapColumnName.LENGTH.value], bins=bins
    )
    h_length_cumsum = np.cumsum(h_length[::-1])[::-1] / sum(h_length)

    for ref, alt in tqdm(ref_cskp_motifs.index.unique()):
        single_substitutions_cskp = single_substitutions_df[
            single_substitutions_df[FeatureMapColumnName.REFERENCE_MOTIF.value].isin(
                ref_cskp_motifs.loc[
                    (ref, alt), FeatureMapColumnName.REFERENCE_MOTIF.value
                ].unique()
            )
            & single_substitutions_df[FeatureMapColumnName.ALTERNATE_MOTIF.value].isin(
                ref_cskp_motifs.loc[
                    (ref, alt), FeatureMapColumnName.ALTERNATE_MOTIF.value
                ].unique()
            )
        ]

        single_substitutions_dna_complement = single_substitutions_df[
            single_substitutions_df[FeatureMapColumnName.REFERENCE_MOTIF.value].isin(
                [
                    revcomp(x)
                    for x in ref_cskp_motifs.loc[
                    (ref, alt), FeatureMapColumnName.REFERENCE_MOTIF.value
                    ].unique()
                ]
            )
            & single_substitutions_df[FeatureMapColumnName.ALTERNATE_MOTIF.value].isin(
                [
                    revcomp(x)
                    for x in ref_cskp_motifs.loc[
                    (ref, alt), FeatureMapColumnName.ALTERNATE_MOTIF.value
                    ].unique()
                ]
            )
        ]

        ref_coverage = coverage_df.loc[
            ref_cskp_motifs.loc[
                (ref, alt), FeatureMapColumnName.REFERENCE_MOTIF.value
            ].unique(),
            COVERAGE,
        ].sum()

        single_substitutions_cskp_hist, _ = np.histogram(
            single_substitutions_cskp[FeatureMapColumnName.INDEX.value],
            bins=bins,
        )
        single_substitutions_dna_comp_hist, _ = np.histogram(
            single_substitutions_dna_complement[FeatureMapColumnName.INDEX.value],
            bins=bins,
        )

        error_rate_results.append(
            pd.DataFrame(
                {
                    ErrorRateColumnName.NAME.value: run_name,
                    ErrorRateColumnName.REF.value: ref,
                    ErrorRateColumnName.ALT.value: alt,
                    ErrorRateColumnName.INDEX.value: bin_centers,
                    ErrorRateColumnName.COUNT_FORWARD.value: single_substitutions_cskp_hist,
                    ErrorRateColumnName.COUNT_REVERSE.value: single_substitutions_dna_comp_hist,
                    ErrorRateColumnName.LENGTH_CDF.value: h_length_cumsum,
                    ErrorRateColumnName.COVERAGE.value: ref_coverage,
                }
            )
        )

    error_rate_results_df = (
        pd.concat(error_rate_results)
        .astype({FeatureMapColumnName.INDEX.value: int})
        .set_index(
            [
                ErrorRateColumnName.NAME.value,
                ErrorRateColumnName.REF.value,
                ErrorRateColumnName.ALT.value,
            ]
        )
    )

    # save the results to parquet file
    error_rate_results_df.to_parquet(
        os.path.join(output_folder, f"{error_per_position_per_motif}.{run_name}{FileExtension.PARQUET.value}")
    )

    return error_rate_results_df


# plot functions #


def _create_plot(run_name: str) -> Tuple:
    fig, axs = plt.subplots(2, 3, figsize=(20, 6), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    artists = [fig.suptitle(f"{run_name}", y=1.05)]
    return fig, axs, artists


def create_graphs(
    error_rate_results,
    run_name,
    figures_folder,
    output_file_prefix: str,
    max_positions_plot: int = MAX_POSITIONS_PLOT,
    position_plot_step_size: int = POSITION_PLOT_STEP_SIZE,
):
    """Create output graphs of positional SNP error rate for each variant"""
    bins = _get_bins(max_positions_plot, position_plot_step_size)

    tight_xlim = error_rate_results[
        error_rate_results.filter(regex="count").max(axis=1) > 10
    ][FeatureMapColumnName.INDEX.value].max()

    # Create basic plot
    fig, axs, artists = _create_plot(run_name)

    for ax, (error_rate_run_name, ref, alt) in zip(
        axs.flatten(), error_rate_results.index.unique()
    ):
        error_rate_plot = error_rate_results.loc[(error_rate_run_name, ref, alt), :]
        plt.sca(ax)
        plt.title(f"{ref}->{alt}", fontsize=FONT_SIZE)

        for column, color, label in [
            (ErrorRateColumnName.COUNT_FORWARD.value, "b", f"{ref}->{alt}"),
            (
                ErrorRateColumnName.COUNT_REVERSE.value,
                "r",
                f"{revcomp(ref)}->{revcomp(alt)}",
            ),
        ]:
            norm = error_rate_plot[ErrorRateColumnName.COVERAGE.value] * (
                error_rate_plot[ErrorRateColumnName.LENGTH_CDF.value]
                / error_rate_plot[ErrorRateColumnName.LENGTH_CDF.value].sum()
            )
            plt.plot(
                error_rate_plot[ErrorRateColumnName.INDEX.value],
                error_rate_plot[column].mask((error_rate_plot[column] < 10)) / norm,
                label=f"{label} = {error_rate_plot[column].sum() / norm.sum():.1E}".replace(
                    "E", r" $\cdot$ "
                ).replace(
                    "-0", "$10^{-"
                )
                + "}$",
                c=color,
                linewidth=2,
            )
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        plt.ylim(0, ax.get_ylim()[1])
        plt.legend(fontsize="medium")

        if ref == "T":
            artists.append(plt.xlabel("Mutation position"))

    plt.xlim(bins[0], bins[-1])
    fig.savefig(
        os.path.join(
            figures_folder,
            f"{output_file_prefix}.{run_name}{FileExtension.PNG.value}",
        ),
        bbox_extra_artists=artists,
        bbox_inches="tight",
    )
    plt.xlim([0, tight_xlim])
    plt.legend(loc="upper right", fontsize="medium")
    fig.savefig(
        os.path.join(
            figures_folder,
            f"{output_file_prefix}.{run_name}.tight_xlim{FileExtension.PNG.value}",
        ),
        bbox_extra_artists=artists,
        bbox_inches="tight",
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="positional_error_rate_profile.py", description=run.__doc__
    )

    parser.add_argument(
        "--featuremap_single_substitutions_dataframe",
        type=str,
        required=True,
        help="""featuremap_single_substitutions_dataframe parquet file""",
    )
    parser.add_argument(
        "--coverage_per_motif",
        type=str,
        required=True,
        help="""coverage_per_motif h5 file""",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Path to which output folder the dataframe will be written (multiple files)""",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        required=False,
        default=DEFAULT_FILE_PREFIX,
        help="""Prefix of output file names for .png files""",
    )
    parser.add_argument(
        "--error_per_pos_file_prefix",
        type=str,
        required=False,
        default=DEFAULT_ERROR_PER_POSITION_PREFIX,
        help="""Prefix of output file names for .png files""",
    )
    parser.add_argument(
        "--flow_order",
        type=str,
        required=False,
        default=DEFAULT_FLOW_ORDER,
        help="""flow order - required for cycle skip annotation """,
    )
    parser.add_argument(
        "--allow_softlclip",
        type=bool,
        required=False,
        default=False,
        help=""" if True: include reads with softclip in the analysis (default False) """,
    )
    parser.add_argument(
        "--position_plot_step_size",
        type=int,
        required=False,
        default=POSITION_PLOT_STEP_SIZE,
        help=""" base pair resolution of output plot """,
    )
    parser.add_argument(
        "--max_position_for_plot",
        type=int,
        required=False,
        default=MAX_POSITIONS_PLOT,
        help=""" max position for plot """,
    )
    parser.add_argument(
        "--edist_threshold",
        type=int,
        required=False,
        default=TH_X_EDIST,
        help=""" Edit distance threshold """,
    )
    parser.add_argument(
        "--xscore_threshold",
        type=int,
        required=False,
        default=TH_X_SCORE,
        help=""" X Score threshold """,
    )

    return parser.parse_args(argv)


def calc_positional_error_rate_profile(  # pylint: disable=too-many-arguments
    single_substitutions_file_name: str,
    coverage_per_motif_file_name: str,
    output_folder: str,
    output_file_prefix: str = DEFAULT_FILE_PREFIX,
    error_per_pos_file_prefix: str = DEFAULT_ERROR_PER_POSITION_PREFIX,
    allow_softlclip: bool = False,
    flow_order: str = DEFAULT_FLOW_ORDER,
    position_plot_step_size: int = POSITION_PLOT_STEP_SIZE,
    max_positions_plot: int = MAX_POSITIONS_PLOT,
    edist_threshold: int = TH_X_EDIST,
    xscore_threshold: int = TH_X_SCORE,
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

    :return: None value return, the process saves created graphs in files and enable plotting them
    """
    # preprocess
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    run_name = preprocess_filename(single_substitutions_file_name)

    # read feature maps
    # TODO: Can these 2 operations be merged?
    single_substitutions_df = load_substitutions(single_substitutions_file_name)
    single_substitutions_df = filter_substitutions_threshold(
        single_substitutions_df, allow_softlclip, edist_threshold, xscore_threshold
    )
    coverage_df = load_coverage_per_motif(coverage_per_motif_file_name, "motif_1")

    # Analysis
    ref_cskp_motifs = get_cycle_skip_motifs(flow_order)

    # process
    error_rate_results = calc_error_rate(
        run_name,
        ref_cskp_motifs,
        single_substitutions_df,
        coverage_df,
        output_folder,
        error_per_pos_file_prefix,
        max_positions_plot,
        position_plot_step_size,
    )

    # Create graphs
    create_graphs(
        error_rate_results,
        run_name,
        output_folder,
        output_file_prefix,
        max_positions_plot,
        position_plot_step_size,
    )


def run(argv: List[str]):
    """ Calculate positional error rate profile for all SNPs """
    print(f"positional_error_rate_profile.run called with {argv}")
    args = parse_args(argv[1:])
    calc_positional_error_rate_profile(
        single_substitutions_file_name=args.featuremap_single_substitutions_dataframe,
        coverage_per_motif_file_name=args.coverage_per_motif,
        output_folder=args.output,
        output_file_prefix=args.output_file_prefix,
        error_per_pos_file_prefix=args.error_per_pos_file_prefix,
        allow_softlclip=args.allow_softlclip,
        flow_order=args.flow_order,
        position_plot_step_size=args.position_plot_step_size,
        max_positions_plot=args.max_position_for_plot,
        edist_threshold=args.edist_threshold,
        xscore_threshold=args.xscore_threshold,
    )
