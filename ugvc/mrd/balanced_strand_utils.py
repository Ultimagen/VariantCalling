from __future__ import annotations

import itertools
import os
import subprocess
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns

from ugvc.utils.metrics_utils import read_sorter_statistics_csv
from ugvc.utils.misc_utils import modify_jupyter_notebook_html, set_pyplot_defaults
from ugvc.vcfbed.variant_annotation import VcfAnnotator

# Display defaults
STRAND_RATIO_AXIS_LABEL = "LIG/HYB strands ratio"


# Supported adapter versions
class BalancedStrandAdapterVersions(Enum):
    LA_v5 = "LA_v5"
    LA_v5and6 = "LA_v5and6"
    LA_v6 = "LA_v6"


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
    A_HMER_END = "ae"
    NATIVE_ADAPTER = "a3"
    STEM_END = "s2"  # when native adapter trimming was done on-tool a modified format is used


class BalancedCategories(Enum):
    # Category names
    MIXED = "MIXED"
    LIG = "LIG"
    HYB = "HYB"
    END_UNREACHED = "END_UNREACHED"
    UNDETERMINED = "UNDETERMINED"


class BalancedCategoriesConsensus(Enum):
    # Category names
    MIXED = "MIXED"
    LIG = "LIG"
    HYB = "HYB"
    DISCORDANT = "DISCORDANT"
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
    STRAND_RATIO_CATEGORY_CONSENSUS = "strand_ratio_category_consensus"


# Input parameter defaults for LAv5+6, LAv5 and LAv6
STRAND_RATIO_LOWER_THRESH = 0.27
STRAND_RATIO_UPPER_THRESH = 0.73
MIN_TOTAL_HMER_LENGTHS_IN_TAGS = 4
MAX_TOTAL_HMER_LENGTHS_IN_TAGS = 8
MIN_STEM_END_MATCHED_LENGTH = 11  # the stem is 12bp, 1 indel allowed as tolerance


class BalancedStrandVcfAnnotator(VcfAnnotator):
    def __init__(
        self,
        adapter_version: str | BalancedStrandAdapterVersions,
        sr_lower: float = STRAND_RATIO_LOWER_THRESH,
        sr_upper: float = STRAND_RATIO_UPPER_THRESH,
        min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
        max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
        min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    ):
        _assert_adapter_version_supported(adapter_version)
        self.adapter_version = (
            adapter_version.value if isinstance(adapter_version, BalancedStrandAdapterVersions) else adapter_version
        )
        self.sr_lower = sr_lower
        self.sr_upper = sr_upper
        self.min_total_hmer_lengths_in_tags = min_total_hmer_lengths_in_tags
        self.max_total_hmer_lengths_in_tags = max_total_hmer_lengths_in_tags
        self.min_stem_end_matched_length = min_stem_end_matched_length

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to add strand ratio and strand ratio category INFO fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            VCF header with strand ratio and strand ratio category INFO fields

        """
        header.add_line(f"##balanced_strand_adapter_version={self.adapter_version}")
        header.add_line(
            f"##python_cmd:add_strand_ratios_and_categories_to_featuremap=MIXED is {self.sr_lower}-{self.sr_upper}"
        )
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
        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """
        Add strand ratio and strand ratio category INFO fields to the VCF records

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of VCF records with strand ratio and strand ratio category INFO fields
        """
        records_out = [None] * len(records)
        for j, record in enumerate(records):
            # get the balanced tags from the VCF record
            balanced_tags = defaultdict(int)
            for x in [v.value for v in TrimmerSegmentTags.__members__.values()]:
                if x in record.info:
                    balanced_tags[x] = int(record.info.get(x))
            # add the start strand ratio and strand ratio category columns to the VCF record
            if self.adapter_version in (
                BalancedStrandAdapterVersions.LA_v5and6.value,
                BalancedStrandAdapterVersions.LA_v5.value,
            ):  # LA_v5 has start tags
                # assign to simple variables for readability
                T_hmer_start = balanced_tags[TrimmerSegmentTags.T_HMER_START.value]
                A_hmer_start = balanced_tags[TrimmerSegmentTags.A_HMER_START.value]
                # determine ratio and category
                tags_sum_start = T_hmer_start + A_hmer_start
                if self.min_total_hmer_lengths_in_tags <= tags_sum_start <= self.max_total_hmer_lengths_in_tags:
                    record.info[HistogramColumnNames.STRAND_RATIO_START.value] = T_hmer_start / (
                        T_hmer_start + A_hmer_start
                    )
                else:
                    record.info[HistogramColumnNames.STRAND_RATIO_START.value] = np.nan
                record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value] = get_strand_ratio_category(
                    record.info[HistogramColumnNames.STRAND_RATIO_START.value],
                    self.sr_lower,
                    self.sr_upper,
                )
            if self.adapter_version in (
                BalancedStrandAdapterVersions.LA_v5and6.value,
                BalancedStrandAdapterVersions.LA_v6.value,
            ):  # LA_v6 has end tags
                # assign to simple variables for readability
                T_hmer_end = balanced_tags[TrimmerSegmentTags.T_HMER_END.value]
                A_hmer_end = balanced_tags[TrimmerSegmentTags.A_HMER_END.value]
                # determine ratio and category
                tags_sum_end = T_hmer_end + A_hmer_end
                # determine if read end was reached
                is_end_reached = (
                    balanced_tags[TrimmerSegmentTags.NATIVE_ADAPTER.value] >= 1
                    or balanced_tags[TrimmerSegmentTags.STEM_END.value] >= self.min_stem_end_matched_length
                )
                record.info[HistogramColumnNames.STRAND_RATIO_END.value] = np.nan
                if not is_end_reached:
                    record.info[
                        HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value
                    ] = BalancedCategories.END_UNREACHED.value
                else:
                    if self.min_total_hmer_lengths_in_tags <= tags_sum_end <= self.max_total_hmer_lengths_in_tags:
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value] = T_hmer_end / (
                            T_hmer_end + A_hmer_end
                        )
                    record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = get_strand_ratio_category(
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value],
                        self.sr_lower,
                        self.sr_upper,
                    )  # this works for nan values as well - returns UNDETERMINED
            records_out[j] = record

        return records_out


# Misc
balanced_category_list = [v.value for v in BalancedCategories.__members__.values()]
supported_adapter_versions = BalancedStrandAdapterVersions.__members__.values()


# pylint: disable=missing-param-doc
def _assert_adapter_version_supported(
    adapter_version: str | BalancedStrandAdapterVersions,
):
    """
    Assert that the adapter version is supported

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check

    Raises
    ------
    AssertionError
        If the adapter version is not supported
    """
    if isinstance(adapter_version, BalancedStrandAdapterVersions):
        assert adapter_version in supported_adapter_versions, (
            f"Unsupported adapter version {adapter_version.value}, "
            + f"supprted values are {', '.join(supported_adapter_versions)}"
        )
    if isinstance(adapter_version, str):
        assert adapter_version in [v.value for v in supported_adapter_versions], (
            f"Unsupported adapter version {adapter_version}, "
            + f"supprted values are {', '.join(supported_adapter_versions)}"
        )


def get_strand_ratio_category(strand_ratio, sr_lower, sr_upper):
    """
    Determine the strand ratio category

    Parameters
    ----------
    strand_ratio : float
        strand ratio
    sr_lower : float
        lower strand ratio threshold for determining strand ratio category MIXED
    sr_upper : float
        upper strand ratio threshold for determining strand ratio category MIXED

    Returns
    -------
    str
        strand ratio category
    """
    if strand_ratio == 0:
        return BalancedCategories.HYB.value
    if strand_ratio == 1:
        return BalancedCategories.LIG.value
    if sr_lower <= strand_ratio <= sr_upper:
        return BalancedCategories.MIXED.value
    return BalancedCategories.UNDETERMINED.value


def read_balanced_strand_trimmer_histogram(
    adapter_version: str | BalancedStrandAdapterVersions,
    trimmer_histogram_csv: str,
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    sample_name: str = "",
    output_filename: str = None,
):
    """
    Read a balanced ePCR trimmer histogram file and add columns for strand ratio and strand ratio category

    Parameters
    ----------
    adapter_version : [str, BalancedStrandAdapterVersions]
        adapter version to check
    trimmer_histogram_csv : str
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
    _assert_adapter_version_supported(adapter_version)
    # read histogram
    df_trimmer_histogram = pd.read_csv(trimmer_histogram_csv)
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
            raise ValueError(f"Missing expected column {col} in {trimmer_histogram_csv}")
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
            f"or {TrimmerSegmentLabels.STEM_END.value} in {trimmer_histogram_csv}"
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
    ].apply(lambda x: get_strand_ratio_category(x, sr_lower, sr_upper))
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
            .apply(lambda x: get_strand_ratio_category(x, sr_lower, sr_upper))
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
    adapter_version: str | BalancedStrandAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group the trimmer histogram by strand ratio category

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns
    """
    # fill end tag with dummy column if it does not exist
    _assert_adapter_version_supported(adapter_version)
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
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value}),
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value)
                .agg({count: "sum"})
                .drop(BalancedCategories.END_UNREACHED.value, errors="ignore")
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
    adapter_version: str | BalancedStrandAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the concordance between the strand ratio categories at the start and end of the read

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns

    Raises
    ------
    ValueError
        If the adapter version is LA_v5 or LA_v6 and the end tag is missing

    """
    _assert_adapter_version_supported(adapter_version)
    if adapter_version in (
        BalancedStrandAdapterVersions.LA_v5,
        BalancedStrandAdapterVersions.LA_v5.value,
        BalancedStrandAdapterVersions.LA_v6,
        BalancedStrandAdapterVersions.LA_v6.value,
    ):
        raise ValueError(
            f"Adapter version {adapter_version} does not have tags on both ends. "
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


def collect_statistics(
    adapter_version: str | BalancedStrandAdapterVersions,
    trimmer_histogram_csv: str,
    sorter_stats_csv: str,
    output_filename: str,
    input_material_ng: float = None,
    **trimmer_histogram_kwargs,
):
    """
    Collect statistics from a balanced ePCR trimmer histogram file and a sorter stats file

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
    trimmer_histogram_csv : str
        path to a balanced strand Trimmer histogram file
    sorter_stats_csv : str
        path to a Sorter stats file
    output_filename : str
        path to save dataframe to in hdf format (should end with .h5)
    input_material_ng : float, optional
        input material in ng, by default None

    Raises
    ------
    ValueError
        If the adapter version is invalid
    """
    _assert_adapter_version_supported(adapter_version)
    # read Trimmer histogram
    df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
        adapter_version, trimmer_histogram_csv, **trimmer_histogram_kwargs
    )
    df_strand_ratio_category = group_trimmer_histogram_by_strand_ratio_category(adapter_version, df_trimmer_histogram)
    adapter_in_both_ends = adapter_version in (
        BalancedStrandAdapterVersions.LA_v5and6,
        BalancedStrandAdapterVersions.LA_v5and6.value,
    )
    if adapter_in_both_ends:
        df_category_concordance = get_strand_ratio_category_concordance(adapter_version, df_trimmer_histogram)
        df_category_concordance_no_end_unreached = df_category_concordance.drop(
            BalancedCategories.END_UNREACHED.value,
            level=HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
            errors="ignore",
        )
        df_category_concordance_no_end_unreached = (
            df_category_concordance_no_end_unreached / df_category_concordance_no_end_unreached.sum().sum()
        )
        df_category_concordance_no_end_unreached = df_category_concordance_no_end_unreached.rename(
            columns={HistogramColumnNames.COUNT.value: HistogramColumnNames.COUNT_NORM.value}
        ).reset_index()
        x = HistogramColumnNames.STRAND_RATIO_CATEGORY_CONSENSUS.value  # otherwise flake8 fails on line length
        df_category_consensus = (
            df_category_concordance_no_end_unreached[
                df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value]
                == df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]
            ]
            .drop(columns=[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value])
            .rename(columns={HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value: x})
            .set_index(x)
        )
        df_category_consensus.loc[
            BalancedCategoriesConsensus.DISCORDANT.value,
            HistogramColumnNames.COUNT_NORM.value,
        ] = (
            1 - df_category_consensus[HistogramColumnNames.COUNT_NORM.value].sum()
        )

    # read Sorter stats
    df_sorter_stats = read_sorter_statistics_csv(sorter_stats_csv)
    if input_material_ng is not None:
        if "Mean_cvg" in df_sorter_stats.index:
            df_sorter_stats.loc["coverage_GE/ng", "value"] = (
                df_sorter_stats.loc["Mean_cvg", "value"] / input_material_ng
            )
        if "PF_Barcode_reads" in df_sorter_stats.index:
            df_sorter_stats.loc["PF_reads/ng", "value"] = (
                df_sorter_stats.loc["PF_Barcode_reads", "value"] / input_material_ng
            )

    # create statistics shortlist
    df_stats_shortlist = df_sorter_stats.reindex(
        [
            "Mean_cvg",
            "coverage_GE/ng",
            "Indel_Rate",
            "Mean_Read_Length",
            "PF_Barcode_reads",
            "PF_reads/ng",
            "% PF_Reads_aligned",
            "% Failed_QC_reads",
            "% Chimeras",
            "% duplicates",
        ]
    )

    if adapter_version in (
        BalancedStrandAdapterVersions.LA_v5,
        BalancedStrandAdapterVersions.LA_v5.value,
    ):
        df_tags = df_strand_ratio_category.drop(BalancedCategories.END_UNREACHED.value, errors="ignore")[
            [HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value]
        ]
        df_tags.index = [f"% {x} reads" for x in df_tags.index]
        df_tags.columns = ["value"]
        df_tags = df_tags * 100 / df_tags.sum()
    elif adapter_version in (
        BalancedStrandAdapterVersions.LA_v6,
        BalancedStrandAdapterVersions.LA_v6.value,
    ):
        df_tags = df_strand_ratio_category[[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]]
        df_tags.index = [f"% {x} reads" for x in df_tags.index]
        df_tags.columns = ["value"]
        df_tags = df_tags * 100 / df_tags.sum()
    elif adapter_version in (
        BalancedStrandAdapterVersions.LA_v5and6,
        BalancedStrandAdapterVersions.LA_v5and6.value,
    ):
        df_tags = df_category_consensus * 100
        df_tags.index = [f"% {x} reads (both tags)" for x in df_tags.index]
        df_tags.columns = ["value"]
        df_strand_ratio_category_norm = (
            df_strand_ratio_category.loc[
                [BalancedCategories.END_UNREACHED.value],
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
            ]
            * 100
            / df_strand_ratio_category[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value].sum()
        ).T.to_frame()
        df_strand_ratio_category_norm.columns = ["value"]
        df_tags = pd.concat((df_tags, df_strand_ratio_category_norm)).rename(
            {
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value: "% end tag not reached",
            }
        )
    else:
        raise ValueError(
            f"Unknown adapter version: {adapter_version if isinstance(adapter_version, str) else adapter_version.value}"
        )

    df_tags.index.name = "metric"
    df_stats_shortlist = pd.concat((df_tags, df_stats_shortlist))

    # save
    if not output_filename.endswith(".h5"):
        output_filename += ".h5"
    with pd.HDFStore(output_filename, "w") as store:
        store["stats_shortlist"] = df_stats_shortlist
        store["sorter_stats"] = df_sorter_stats
        store["trimmer_histogram"] = df_trimmer_histogram
        store["strand_ratio_category_counts"] = df_strand_ratio_category
        store["strand_ratio_category_norm"] = df_strand_ratio_category / df_strand_ratio_category.sum()
        if adapter_in_both_ends:
            store["df_category_concordance"] = df_category_concordance
            store["df_category_consensus"] = df_category_consensus


def add_strand_ratios_and_categories_to_featuremap(
    adapter_version: str | BalancedStrandAdapterVersions,
    input_featuremap_vcf: str,
    output_featuremap_vcf: str,
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    chunk_size: int = 10000,
    multiprocess_contigs: bool = False,
):
    """
    Add strand ratio and strand ratio category columns to a featuremap VCF file

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
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
    chunk_size : int, optional
            The chunk size. Defaults to 10000.
    multiprocess_contigs : bool, optional
        If True, runs in parallel over different contigs. Defaults to False.
    """
    _assert_adapter_version_supported(adapter_version)
    balanced_strand_variant_annotator = BalancedStrandVcfAnnotator(
        adapter_version=adapter_version,
        sr_lower=sr_lower,
        sr_upper=sr_upper,
        min_total_hmer_lengths_in_tags=min_total_hmer_lengths_in_tags,
        max_total_hmer_lengths_in_tags=max_total_hmer_lengths_in_tags,
        min_stem_end_matched_length=min_stem_end_matched_length,
    )
    BalancedStrandVcfAnnotator.process_vcf(
        annotators=[balanced_strand_variant_annotator],
        input_path=input_featuremap_vcf,
        output_path=output_featuremap_vcf,
        chunk_size=chunk_size,
        multiprocess_contigs=multiprocess_contigs,
    )


def plot_balanced_strand_ratio(
    adapter_version: str | BalancedStrandAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the strand ratio histogram

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    """
    _assert_adapter_version_supported(adapter_version)
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
    for sr, sr_category, sr_category_hist, label in zip(
        (
            HistogramColumnNames.STRAND_RATIO_START.value,
            HistogramColumnNames.STRAND_RATIO_END.value,
        ),
        (
            HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
            HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        ),
        (
            HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
            HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value,
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
                adapter_version, df_trimmer_histogram
            )
            mixed_reads_ratio = (
                df_trimmer_histogram_by_strand_ratio_category.loc[BalancedCategories.MIXED.value, sr_category_hist]
                / df_trimmer_histogram_by_strand_ratio_category[sr_category_hist].sum()
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
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
    return ax


def plot_strand_ratio_category(
    adapter_version: str | BalancedStrandAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the strand ratio category histogram

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
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
    _assert_adapter_version_supported(adapter_version)
    # display settings
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(14, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)

    # group by category

    df_trimmer_histogram_by_strand_ratio_category = (
        (group_trimmer_histogram_by_strand_ratio_category(adapter_version, df_trimmer_histogram))
        .drop(
            columns=[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value],
            errors="ignore",
        )
        .rename(
            columns={
                HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value: "Start tag strand ratio category",
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value: "End tag strand ratio category",
            }
        )
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
        ax.bar_label(cont, labels=[f"{x:.1%}" for x in cont.datavalues], label_type="edge")
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
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
    return ax


def plot_strand_ratio_category_concordnace(
    adapter_version: str | BalancedStrandAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    axs: list[plt.Axes] = None,
) -> list[plt.Axes]:
    """
    Plot the strand ratio category concordance heatmap

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
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
    _assert_adapter_version_supported(adapter_version)
    # get concordance
    df_category_concordance = get_strand_ratio_category_concordance(adapter_version, df_trimmer_histogram)

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
        df_plot.index.name = "Start tag category"
        df_plot.columns.name = "End tag category"

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
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )
    return axs


# pylint: disable=differing-param-doc
# pylint: disable=differing-type-doc
def plot_trimmer_histogram(
    adapter_version: str | BalancedStrandAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
) -> list[plt.Axes]:
    """
    Plot the trimmer hmer calls on a heatmap

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_balanced_strand_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    axs : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8

    Returns
    -------
    axs: list[plt.Axes]
        list of axes objects to which the output was plotted

    """
    _assert_adapter_version_supported(adapter_version)
    # display settings
    set_pyplot_defaults()
    # generate axs
    if adapter_version in (
        BalancedStrandAdapterVersions.LA_v5and6,
        BalancedStrandAdapterVersions.LA_v5and6.value,
    ):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.3)
        plot_iter = (
            (
                TrimmerSegmentLabels.A_HMER_START.value,
                TrimmerSegmentLabels.T_HMER_START.value,
                HistogramColumnNames.COUNT_NORM.value,
                "Start tag",
            ),
            (
                TrimmerSegmentLabels.A_HMER_END.value,
                TrimmerSegmentLabels.T_HMER_END.value,
                HistogramColumnNames.COUNT_NORM.value,
                "End tag",
            ),
        )
    else:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        axs = [axs]
        if adapter_version in (
            BalancedStrandAdapterVersions.LA_v5,
            BalancedStrandAdapterVersions.LA_v5.value,
        ):
            plot_iter = (
                (
                    TrimmerSegmentLabels.A_HMER_START.value,
                    TrimmerSegmentLabels.T_HMER_START.value,
                    HistogramColumnNames.COUNT_NORM.value,
                    "Start tag",
                ),
            )
        elif adapter_version in (
            BalancedStrandAdapterVersions.LA_v6,
            BalancedStrandAdapterVersions.LA_v6.value,
        ):
            plot_iter = (
                (
                    TrimmerSegmentLabels.A_HMER_END.value,
                    TrimmerSegmentLabels.T_HMER_END.value,
                    HistogramColumnNames.COUNT_NORM.value,
                    "End tag",
                ),
            )

    # plot
    title_handle = plt.suptitle(title, y=1.03)
    for ax, (xcol, ycol, zcol, subtitle) in zip(axs, plot_iter):
        # group by strand ratio and strand ratio category for non-undetermined reads
        df_plot = df_trimmer_histogram.groupby([xcol, ycol]).agg({zcol: "sum"}).reset_index()
        df_hmer_sum = df_plot[[xcol, ycol]].sum(axis=1)
        df_plot = df_plot[
            (min_total_hmer_lengths_in_tags <= df_hmer_sum) & (df_hmer_sum <= max_total_hmer_lengths_in_tags)
        ]
        df_plot.loc[:, zcol] = df_plot[zcol] / df_plot[zcol].sum()
        # plot
        plt.sca(ax)
        plt.scatter(df_plot[xcol], df_plot[ycol], s=500 * df_plot[zcol], c=df_plot[zcol])
        plt.colorbar()
        plt.xticks(range(int(plt.gca().get_xlim()[1]) + 1))
        plt.yticks(range(int(plt.gca().get_ylim()[1]) + 1))
        plt.xlabel(xcol.replace("_", " "))
        plt.ylabel(ycol.replace("_", " "))
        plt.title(subtitle, fontsize=22)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )
    return axs


# pylint: disable=too-many-arguments
def balanced_strand_analysis(
    adapter_version: str | BalancedStrandAdapterVersions,
    trimmer_histogram_csv: str,
    sorter_stats_csv: str,
    output_path: str,
    output_basename: str = None,
    collect_statistics_kwargs: dict = None,
    generate_report: bool = True,
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
):
    """
    Run the balanced strand analysis pipeline

    Parameters
    ----------
    adapter_version : str | BalancedStrandAdapterVersions
        adapter version to check
    trimmer_histogram_csv : str
        path to a balanced strand Trimmer histogram file
    sorter_stats_csv : str
        path to a Sorter stats file
    output_path : str
        path to save data and report to
    output_basename : str, optional
        basename for output files, by default None (basename of trimmer_histogram_csv)
    collect_statistics_kwargs : dict, optional
        kwargs for collect_statistics, by default None
    generate_report
        if True, generate an html+jupyter report, by default True
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
    # check inputs
    _assert_adapter_version_supported(adapter_version)
    assert os.path.isfile(trimmer_histogram_csv), f"{trimmer_histogram_csv} not found"
    assert os.path.isfile(sorter_stats_csv), f"{sorter_stats_csv} not found"

    # make output directory and determine base file name
    os.makedirs(output_path, exist_ok=True)
    if output_basename is None:
        output_basename = os.path.basename(trimmer_histogram_csv)
    output_statistics_file = os.path.join(output_path, f"{output_basename}.statistics.h5")
    output_trimmer_histogram_plot = os.path.join(output_path, f"{output_basename}.trimmer_histogram.png")
    output_strand_ratio_plot = os.path.join(output_path, f"{output_basename}.strand_ratio.png")
    output_strand_ratio_category_plot = os.path.join(output_path, f"{output_basename}.strand_ratio_category.png")
    output_strand_ratio_category_concordance_plot = os.path.join(
        output_path, f"{output_basename}.strand_ratio_category_concordance.png"
    )
    output_report_ipynb = os.path.join(output_path, f"{output_basename}.balanced_strand_qc_report.ipynb")
    output_report_html = os.path.join(output_path, f"{output_basename}.balanced_strand_qc_report.html")

    # collect statistics
    # create the input for collect statistics
    if collect_statistics_kwargs is None:
        collect_statistics_kwargs = {}
    collect_statistics_kwargs.setdefault("output_filename", output_statistics_file)
    collect_statistics_kwargs.setdefault("adapter_version", adapter_version)
    collect_statistics_kwargs.setdefault("trimmer_histogram_csv", trimmer_histogram_csv)
    collect_statistics_kwargs.setdefault("sorter_stats_csv", sorter_stats_csv)
    collect_statistics_kwargs.setdefault("sr_lower", sr_lower)
    collect_statistics_kwargs.setdefault("sr_upper", sr_upper)
    collect_statistics_kwargs.setdefault("min_total_hmer_lengths_in_tags", min_total_hmer_lengths_in_tags)
    collect_statistics_kwargs.setdefault("max_total_hmer_lengths_in_tags", max_total_hmer_lengths_in_tags)
    collect_statistics_kwargs.setdefault("min_stem_end_matched_length", min_stem_end_matched_length)
    collect_statistics(**collect_statistics_kwargs)

    # read Trimmer histogram output from collect statistics
    df_trimmer_histogram = pd.read_hdf(output_statistics_file, "trimmer_histogram")

    # generate plots
    plot_trimmer_histogram(
        adapter_version,
        df_trimmer_histogram,
        title=f"{output_basename} hmer calls",
        output_filename=output_trimmer_histogram_plot,
    )
    plot_balanced_strand_ratio(
        adapter_version,
        df_trimmer_histogram,
        title=f"{output_basename} strand ratio",
        output_filename=output_strand_ratio_plot,
    )
    plot_strand_ratio_category(
        adapter_version,
        df_trimmer_histogram,
        title=f"{output_basename} strand ratio category",
        output_filename=output_strand_ratio_category_plot,
    )
    if adapter_version in (
        BalancedStrandAdapterVersions.LA_v5and6,
        BalancedStrandAdapterVersions.LA_v5and6.value,
    ):
        plot_strand_ratio_category_concordnace(
            adapter_version,
            df_trimmer_histogram,
            title=f"{output_basename} strand ratio category concordance",
            output_filename=output_strand_ratio_category_concordance_plot,
        )

    # generate report
    if generate_report:
        template_notebook = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "reports/balanced_strand/balanced_strand_qc_report.ipynb",
        )
        illustration_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "reports/balanced_strand/balanced_strand_sequencing_illustration.png",
        )
        parameters = dict(
            adapter_version=adapter_version if isinstance(adapter_version, str) else adapter_version.value,
            statistics_h5=output_statistics_file,
            trimmer_histogram_png=output_trimmer_histogram_plot,
            strand_ratio_png=output_strand_ratio_plot,
            strand_ratio_category_png=output_strand_ratio_category_plot,
            sr_lower=sr_lower,
            sr_upper=sr_upper,
            min_total_hmer_lengths_in_tags=min_total_hmer_lengths_in_tags,
            max_total_hmer_lengths_in_tags=max_total_hmer_lengths_in_tags,
            illustration_file=illustration_file,
        )
        if adapter_version in (
            BalancedStrandAdapterVersions.LA_v5and6,
            BalancedStrandAdapterVersions.LA_v5and6.value,
        ):
            parameters["strand_ratio_category_concordance_png"] = output_strand_ratio_category_concordance_plot
        # inject parameters and run notebook
        papermill_params = f"{' '.join([f'-p {k} {v}' for k, v in parameters.items()])}"
        papermill_cmd = f"papermill {template_notebook} {output_report_ipynb} {papermill_params} -k python3"
        subprocess.check_call(papermill_cmd.split())
        # convert to html
        subprocess.check_call(f"jupyter nbconvert {output_report_ipynb} --to html --no-input".split())
        # edit html for readability
        modify_jupyter_notebook_html(output_report_html)
