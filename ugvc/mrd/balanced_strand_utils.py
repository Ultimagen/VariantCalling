from __future__ import annotations

import itertools
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns

from ugvc.utils.misc_utils import set_pyplot_defaults


# Trimmer segment labels and tags
class TrimmerSegmentLabels(Enum):
    T_HMER_START = "T_hmer_start"
    T_HMER_END = "T_hmer_end"
    A_HMER_START = "A_hmer_start"
    A_HMER_END = "A_hmer_end"
    NATIVE_ADAPTER = "native_adapter_with_leading_C"
    STEM_END = "Stem_end"  # when native adapter trimming was done on-tool a modified format is used


class TrimmerSegmentTags(Enum):
    T_HMER_START = "ts"
    T_HMER_END = "te"
    A_HMER_START = "as"
    A_HMER_END = "te"
    NATIVE_ADAPTER = "a3"
    STEM_END = "s2"  # when native adapter trimming was done on-tool a modified format is used


class BalancedCategories(Enum):
    # Category names
    MIXED = "MIXED"
    LIG = "LIG"
    HYB = "HYB"
    END_UNREACHED = "END_UNREACHED"
    UNDETERMINED = "UNDETERMINED"


class HistogramColumnNames(Enum):
    # Internal names
    COUNT = "count"
    COUNT_NORM = "count_norm"
    STRAND_RATIO_START = "strand_ratio_start"
    STRAND_RATIO_END = "strand_ratio_end"
    STRAND_RATIO_CATEGORY_START = "strand_ratio_category_start"
    STRAND_RATIO_CATEGORY_END = "strand_ratio_category_end"
    STRAND_RATIO_CATEGORY_END_NO_UNREACHED = "strand_ratio_category_end_no_unreached"


# Input parameter defaults
STRAND_RATIO_LOWER_THRESH = 0.27
STRAND_RATIO_UPPER_THRESH = 0.73
MIN_TOTAL_HMER_LENGTHS_IN_TAGS = 4
MAX_TOTAL_HMER_LENGTHS_IN_TAGS = 8
MIN_STEM_END_MATCHED_LENGTH = 11  # the stem is 12bp, 1 indel allowed as tolerance

# Display defaults
STRAND_RATIO_AXIS_LABEL = "LIG/HYB strands ratio"
balanced_category_list = [v.value for v in BalancedCategories.__members__.values()]


def get_annotation(x, sr_lower, sr_upper):
    if x == 0:
        return BalancedCategories.HYB.value
    if x == 1:
        return BalancedCategories.LIG.value
    if sr_lower <= x <= sr_upper:
        return BalancedCategories.MIXED.value
    return BalancedCategories.UNDETERMINED.value


def read_balanced_strand_trimmer_histogram(
    trimmer_histogram,
    sr_lower=STRAND_RATIO_LOWER_THRESH,
    sr_upper=STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags=MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags=MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
    min_stem_end_matched_length=MIN_STEM_END_MATCHED_LENGTH,
    sample_name="",
    output_filename=None,
):
    """
    Read a balanced ePCR trimmer histogram file and add columns for strand ratio and strand ratio category

    Parameters
    ----------
    trimmer_histogram : str
        path to a balanced ePCR trimmer histogram file
    sr_lower : float, optional
        lower strand ratio threshold for determining strand ratio category
        default 0.27
    sr_upper : float, optional
        upper strand ratio threshold for determining strand ratio category
        default 0.73
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8
    min_stem_end_matched_length : int, optional
        minimum length of stem end matched to determine the read end was reached
    sample_name : str, optional
        sample name to use as index, by default ""
    output_filename : str, optional
        path to save dataframe to in parquet format, by default None (not saved).

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio and strand ratio category columns

    Raises
    ------
    ValueError
        If required columns are missing ("count", "T_hmer_start", "A_hmer_start")
    """
    # read histogram
    df_trimmer_histogram = pd.read_csv(trimmer_histogram)
    # change legacy segment names
    df_trimmer_histogram = df_trimmer_histogram.rename(
        columns={
            "T hmer": TrimmerSegmentLabels.T_HMER_START.value,
            "A hmer": TrimmerSegmentLabels.A_HMER_START.value,
            "A_hmer_5": TrimmerSegmentLabels.A_HMER_START.value,
            "T_hmer_5": TrimmerSegmentLabels.T_HMER_START.value,
            "A_hmer_3": TrimmerSegmentLabels.A_HMER_END.value,
            "T_hmer_3": TrimmerSegmentLabels.T_HMER_END.value,
        }
    )
    # make sure expected columns exist
    for col in (
        HistogramColumnNames.COUNT.value,
        TrimmerSegmentLabels.T_HMER_START.value,
        TrimmerSegmentLabels.A_HMER_START.value,
    ):
        if col not in df_trimmer_histogram.columns:
            raise ValueError(f"Missing expected column {col} in {trimmer_histogram}")
    if (
        TrimmerSegmentLabels.A_HMER_END.value in df_trimmer_histogram.columns
        or TrimmerSegmentLabels.T_HMER_END.value in df_trimmer_histogram.columns
    ) and (
        TrimmerSegmentLabels.NATIVE_ADAPTER.value not in df_trimmer_histogram.columns
        and TrimmerSegmentLabels.STEM_END.value not in df_trimmer_histogram.columns
    ):
        # If an end tag exists (LA-v6)
        raise ValueError(
            f"Missing expected column {TrimmerSegmentLabels.NATIVE_ADAPTER.value} "
            f"or {TrimmerSegmentLabels.STEM_END.value} in {trimmer_histogram}"
        )

    df_trimmer_histogram.index.name = sample_name

    # add normalized count column
    df_trimmer_histogram = df_trimmer_histogram.assign(
        count_norm=df_trimmer_histogram[HistogramColumnNames.COUNT.value]
        / df_trimmer_histogram[HistogramColumnNames.COUNT.value].sum()
    )
    # add strand ratio columns and determine categories
    tags_sum_start = (
        df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_START.value]
        + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_START.value]
    )
    df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_START.value] = (
        (df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_START.value] / tags_sum_start)
        .where((tags_sum_start >= min_total_hmer_lengths_in_tags) & (tags_sum_start <= max_total_hmer_lengths_in_tags))
        .round(2)
    )
    # determine strand ratio category
    df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value] = df_trimmer_histogram[
        HistogramColumnNames.STRAND_RATIO_START.value
    ].apply(lambda x: get_annotation(x, sr_lower, sr_upper))
    if (
        TrimmerSegmentLabels.A_HMER_END.value in df_trimmer_histogram.columns
        or TrimmerSegmentLabels.T_HMER_END.value in df_trimmer_histogram.columns
    ):
        # if only one of the end tags exists (maybe a small subsample) assign the other to 0
        for c in (
            TrimmerSegmentLabels.A_HMER_END.value,
            TrimmerSegmentLabels.T_HMER_END.value,
        ):
            if c not in df_trimmer_histogram.columns:
                df_trimmer_histogram.loc[:, c] = 0

        tags_sum_end = (
            df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_END.value]
            + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_END.value]
        )
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_END.value] = (
            (
                df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_END.value]
                / (
                    df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_END.value]
                    + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_END.value]
                )
            )
            .where((tags_sum_end >= min_total_hmer_lengths_in_tags) & (tags_sum_end <= max_total_hmer_lengths_in_tags))
            .round(2)
        )
        # determine if end was reached - at least 1bp native adapter or all of the end stem were found
        is_end_reached = (
            df_trimmer_histogram[TrimmerSegmentLabels.NATIVE_ADAPTER.value] >= 1
            if TrimmerSegmentLabels.NATIVE_ADAPTER.value in df_trimmer_histogram.columns
            else df_trimmer_histogram[TrimmerSegmentLabels.STEM_END.value] >= min_stem_end_matched_length
        )
        # determine strand ratio category
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = (
            df_trimmer_histogram[HistogramColumnNames.STRAND_RATIO_END.value]
            .apply(lambda x: get_annotation(x, sr_lower, sr_upper))
            .where(is_end_reached, BalancedCategories.END_UNREACHED.value)
        )

    # assign normalized column
    df_trimmer_histogram = df_trimmer_histogram.assign(
        **{
            HistogramColumnNames.COUNT_NORM.value: df_trimmer_histogram[HistogramColumnNames.COUNT.value]
            / df_trimmer_histogram[HistogramColumnNames.COUNT.value].sum()
        }
    )

    # save to parquet
    if output_filename is not None:
        df_trimmer_histogram.to_parquet(output_filename)

    return df_trimmer_histogram


def group_trimmer_histogram_by_strand_ratio_category(
    df_trimmer_histogram: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group the trimmer histogram by strand ratio category

    Parameters
    ----------
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns
    """
    # fill end tag with dummy column if it does not exist
    if HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value not in df_trimmer_histogram.columns:
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = np.nan
    # Group by strand ratio category
    count = HistogramColumnNames.COUNT.value
    df_trimmer_histogram_by_category = (
        pd.concat(
            (
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value)
                .agg({count: "sum"})
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value}),
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value)
                .agg({count: "sum"})
                .drop(BalancedCategories.END_UNREACHED.value, errors="ignore")
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value}),
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value)
                .agg({count: "sum"})
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value}),
            ),
            axis=1,
        )
        .reindex(balanced_category_list)
        .dropna(how="all", axis=1)
        .fillna(0)
        .astype(int)
    )
    # drop dummy column
    if HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value not in df_trimmer_histogram.columns:
        df_trimmer_histogram_by_category = df_trimmer_histogram_by_category.drop(
            [
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value,
            ],
            axis=1,
        )
    return df_trimmer_histogram_by_category


def get_strand_ratio_category_concordance(
    df_trimmer_histogram: pd.DataFrame,
) -> pd.DataFrame:
    if HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value not in df_trimmer_histogram:
        raise ValueError(
            f"Missing expected column {HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value}."
            "Cannot calculate strand tag category concordance."
        )
    df_category_concordance = (
        df_trimmer_histogram.groupby(
            [
                HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
            ]
        )
        .agg({HistogramColumnNames.COUNT.value: "sum"})
        .reindex(
            itertools.product(
                [v for v in balanced_category_list if v != BalancedCategories.END_UNREACHED.value],
                balanced_category_list,
            )
        )
        .fillna(0)
    )
    df_category_concordance = df_category_concordance / df_category_concordance.sum().sum()
    return df_category_concordance


def add_strand_ratios_and_categories_to_featuremap(
    input_featuremap_vcf: str,
    output_featuremap_vcf: str,
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
):
    """
    Add strand ratio and strand ratio category columns to a featuremap VCF file

    Parameters
    ----------
    input_featuremap_vcf : str
        path to input featuremap VCF file
    output_featuremap_vcf : str
        path to which the output featuremap VCF with the additional fields will be written
    sr_lower : float, optional
        lower strand ratio threshold for determining strand ratio category
        default 0.27
    sr_upper : float, optional
        upper strand ratio threshold for determining strand ratio category
        default 0.73
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8
    min_stem_end_matched_length : int, optional
        minimum length of stem end matched to determine the read end was reached
    """

    # iterate over the VCF file and add the strand ratio and strand ratio category columns
    with pysam.VariantFile(input_featuremap_vcf) as input_vcf:
        header = input_vcf.header
        header.add_line(f"##python_cmd:add_strand_ratios_and_categories_to_featuremap=MIXED is {sr_lower}-{sr_upper}")
        header.add_line(
            f"##INFO=<ID={HistogramColumnNames.STRAND_RATIO_START.value},"
            'Number=1,Type=Float,Description="Ratio of LIG and HYB strands '
            'measured from the tag in the start of the read">'
        )
        header.add_line(
            f"##INFO=<ID={HistogramColumnNames.STRAND_RATIO_END.value},"
            'Number=1,Type=Float,Description="Ratio of LIG and HYB strands '
            'measured from the tag in the end of the read">'
        )
        header.add_line(
            f"##INFO=<ID={HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value},"
            'Number=1,Type=String,Description="Balanced read category derived from the ratio of LIG and HYB strands '
            "measured from the tag in the start of the read, options: "
            f'{", ".join(balanced_category_list)}">'
        )
        header.add_line(
            f"##INFO=<ID={HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value},"
            'Number=1,Type=String,Description="Balanced read category derived from the ratio of LIG and HYB strands '
            "measured from the tag in the end of the read, options: "
            f'{", ".join(balanced_category_list)}">'
        )
        with pysam.VariantFile(output_featuremap_vcf, "w", header=header) as output_vcf:
            for record in input_vcf:
                # get the balanced tags from the VCF record
                balanced_tags = defaultdict(int)
                for x in [v.value for v in TrimmerSegmentTags.__members__.values()]:
                    if x in record.info:
                        balanced_tags[x] = int(record.info.get(x))
                # add the start strand ratio and strand ratio category columns to the VCF record
                if (
                    TrimmerSegmentTags.A_HMER_START.value in balanced_tags
                    or TrimmerSegmentTags.T_HMER_START.value in balanced_tags
                ):
                    # assign to simple variables for readability
                    T_hmer_start = balanced_tags[TrimmerSegmentTags.T_HMER_START.value]
                    A_hmer_start = balanced_tags[TrimmerSegmentTags.A_HMER_START.value]
                    # determine ratio and category
                    tags_sum_start = T_hmer_start + A_hmer_start
                    if min_total_hmer_lengths_in_tags <= tags_sum_start <= max_total_hmer_lengths_in_tags:
                        record.info[HistogramColumnNames.STRAND_RATIO_START.value] = T_hmer_start / (
                            T_hmer_start + A_hmer_start
                        )
                    else:
                        record.info[HistogramColumnNames.STRAND_RATIO_START.value] = np.nan
                    record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value] = get_annotation(
                        record.info[HistogramColumnNames.STRAND_RATIO_START.value],
                        sr_lower,
                        sr_upper,
                    )
                if (
                    TrimmerSegmentTags.A_HMER_END.value in balanced_tags
                    or TrimmerSegmentTags.T_HMER_END.value in balanced_tags
                ):
                    # assign to simple variables for readability
                    T_hmer_end = balanced_tags[TrimmerSegmentTags.T_HMER_END.value]
                    A_hmer_end = balanced_tags[TrimmerSegmentTags.A_HMER_END.value]
                    # determine ratio and category
                    tags_sum_end = T_hmer_end + A_hmer_end
                    # determine if read end was reached
                    is_end_reached = (
                        balanced_tags[TrimmerSegmentTags.NATIVE_ADAPTER.value] >= 1
                        if TrimmerSegmentTags.NATIVE_ADAPTER.value in balanced_tags
                        else balanced_tags[TrimmerSegmentTags.STEM_END.value] >= min_stem_end_matched_length
                    )
                    if not is_end_reached:
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value] = np.nan
                        record.info[
                            HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value
                        ] = BalancedCategories.END_UNREACHED.value
                    elif (
                        min_total_hmer_lengths_in_tags <= tags_sum_end <= max_total_hmer_lengths_in_tags
                    ) and is_end_reached:
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value] = T_hmer_end / (
                            T_hmer_end + A_hmer_end
                        )
                    else:
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value] = np.nan
                    record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = get_annotation(
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value],
                        sr_lower,
                        sr_upper,
                    )  # this works for nan values as well - returns UNDETERMINED
                # write the updated VCF record to the output VCF file
                output_vcf.write(record)


def plot_balanced_strand_ratio(
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the strand ratio histogram

    Parameters
    ----------
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    """
    # display settings
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)

    ylim_max = 0
    colors = {
        HistogramColumnNames.STRAND_RATIO_START.value: "xkcd:royal blue",
        HistogramColumnNames.STRAND_RATIO_END.value: "xkcd:red orange",
    }
    markers = {
        HistogramColumnNames.STRAND_RATIO_START.value: "o",
        HistogramColumnNames.STRAND_RATIO_END.value: "s",
    }
    # plot strand ratio histograms for both start and end tags
    for sr, sr_category, label in zip(
        (
            HistogramColumnNames.STRAND_RATIO_START.value,
            HistogramColumnNames.STRAND_RATIO_END.value,
        ),
        (
            HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
            HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        ),
        ("Start tag", "End tag"),
    ):
        if sr in df_trimmer_histogram.columns:
            # group by strand ratio and strand ratio category for non-undetermined reads
            df_plot = (
                df_trimmer_histogram.sort_values(HistogramColumnNames.COUNT.value, ascending=False)
                .dropna(subset=[sr])
                .groupby(sr)
                .agg({HistogramColumnNames.COUNT_NORM.value: "sum", sr_category: "first"})
                .reset_index()
            )
            # normalize by the total number of reads where the end was reached
            total_reads_norm_count_with_end_reached = df_plot.query(
                f"({sr_category} != '{BalancedCategories.END_UNREACHED}') and "
                f"({sr_category} != '{BalancedCategories.UNDETERMINED}')"
            )[HistogramColumnNames.COUNT_NORM.value].sum()
            y = df_plot[HistogramColumnNames.COUNT_NORM.value] / total_reads_norm_count_with_end_reached
            # get category counts
            df_trimmer_histogram_by_strand_ratio_category = group_trimmer_histogram_by_strand_ratio_category(
                df_trimmer_histogram
            )
            mixed_reads_ratio = (
                df_trimmer_histogram_by_strand_ratio_category.loc[BalancedCategories.MIXED.value, sr_category]
                / df_trimmer_histogram_by_strand_ratio_category[sr_category].sum()
            )
            # plot
            plt.plot(
                df_plot[sr],
                y,
                "-",
                c=colors[sr],
                marker=markers[sr],
                label=f"{label}: {mixed_reads_ratio:.1%}" " mixed reads",
            )
            ylim_max = max(ylim_max, y.max() + 0.07)
    legend_handle = plt.legend(loc="upper left", fontsize=14, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=24)

    plt.xlabel(STRAND_RATIO_AXIS_LABEL)
    plt.ylabel("Relative abundance", fontsize=20)
    plt.ylim(0, ylim_max)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
    return ax


def plot_strand_ratio_category(
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the strand ratio category histogram

    Parameters
    ----------
    df_trimmer_histogram : pd.DataFrame
            dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    Returns
    -------
        matplotlib.axes.Axes

    """
    # display settings
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(14, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)

    # group by category

    df_trimmer_histogram_by_strand_ratio_category = group_trimmer_histogram_by_strand_ratio_category(
        df_trimmer_histogram
    )
    df_plot = (
        (df_trimmer_histogram_by_strand_ratio_category / df_trimmer_histogram_by_strand_ratio_category.sum())
        .reset_index()
        .melt(id_vars="index", var_name="")
    )
    # plot
    sns.barplot(
        data=df_plot,
        x="index",
        y="value",
        hue="",
        ax=ax,
    )
    for cont in ax.containers:
        ax.bar_label(cont, fmt="{:.1%}", label_type="edge")
    plt.xticks(rotation=10, ha="center")
    plt.xlabel("")
    plt.ylabel("Relative abundance", fontsize=22)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + 0.04)
    legend_handle = plt.legend(bbox_to_anchor=(1.01, 1), fontsize=14, framealpha=0.95)
    title_handle = plt.title(title)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
    return ax


def plot_strand_ratio_category_concordnace(
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    axs: list[plt.Axes] = None,
) -> list[plt.Axes]:
    """
    Plot the strand ratio category concordance heatmap

    Parameters
    ----------
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    axs : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    Returns
    -------
    list of axes objects to which the output was plotted

    """

    # get concordance
    df_category_concordance = get_strand_ratio_category_concordance(df_trimmer_histogram)

    df_category_concordance_no_end_unreached = df_category_concordance.drop(
        BalancedCategories.END_UNREACHED.value,
        level=HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        errors="ignore",
    )
    df_category_concordance_no_end_unreached = (
        df_category_concordance_no_end_unreached / df_category_concordance_no_end_unreached.sum().sum()
    )
    # display settings
    set_pyplot_defaults()
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.subplots_adjust(hspace=0.7)
    plt.suptitle(title)
    # plot
    for ax, subtitle, df_plot in zip(
        axs,
        ("All reads", "Only reads where end was reached"),
        (df_category_concordance, df_category_concordance_no_end_unreached),
    ):
        df_plot = df_plot.unstack().droplevel(0, axis=1)
        df_plot = df_plot.loc[
            [v for v in balanced_category_list if v != BalancedCategories.END_UNREACHED.value],
            [v for v in balanced_category_list if v in df_plot.columns],
        ].fillna(0)
        df_plot.index.name = "Start tag catergory"
        df_plot.columns.name = "End tag catergory"

        if df_plot.shape[0] == 0:
            continue

        sns.heatmap(
            df_plot,
            annot=True,
            fmt=".1%",
            cmap="rocket",
            linewidths=4,
            linecolor="white",
            cbar=False,
            ax=ax,
            annot_kws={"size": 18},
        )
        ax.grid(False)
        plt.sca(ax)
        plt.xticks(rotation=20)
        title_handle = ax.set_title(subtitle, fontsize=20)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )
