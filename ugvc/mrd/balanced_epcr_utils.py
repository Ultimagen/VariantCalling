from enum import Enum

import pandas as pd


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
    END_UNREACHED = "END_UNREACHED"
    HYB = "HYB"
    LIG = "LIG"
    MIXED = "MIXED"
    UNDETERMINED = "UNDETERMINED"


class HistogramColumnNames(Enum):
    # Internal names
    COUNT = "count"
    STRAND_RATIO_START = "strand_ratio_start"
    STRAND_RATIO_END = "strand_ratio_end"
    STRAND_RATIO_CATEGORY_START = "strand_ratio_category_start"
    STRAND_RATIO_CATEGORY_END = "strand_ratio_category_end"


# Input parameter defaults
STRAND_RATIO_LOWER_THRESH = 0.27
STRAND_RATIO_UPPER_THRESH = 0.73
MIN_TOTAL_HMER_LENGTHS_IN_TAGS = 4
MAX_TOTAL_HMER_LENGTHS_IN_TAGS = 8
MIN_STEM_END_MATCHED_LENGTH = 11  # the stem is 12bp, 1 indel allowed as tolerance


def get_annotation(x, sr_lower, sr_upper):
    if x == 0:
        return BalancedCategories.HYB.value
    if x == 1:
        return BalancedCategories.LIG.value
    if sr_lower <= x <= sr_upper:
        return BalancedCategories.MIXED.value
    return BalancedCategories.UNDETERMINED.value


def read_balanced_epcr_trimmer_histogram(
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
    tags_sum_5 = (
        df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_START.value]
        + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_START.value]
    )
    df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_START.value] = (
        (df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_START.value] / (tags_sum_5))
        .where((tags_sum_5 >= min_total_hmer_lengths_in_tags) & (tags_sum_5 <= max_total_hmer_lengths_in_tags))
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
        for c in (TrimmerSegmentLabels.A_HMER_END.value, TrimmerSegmentLabels.T_HMER_END.value):
            if c not in df_trimmer_histogram.columns:
                df_trimmer_histogram.loc[:, c] = 0

        tags_sum_3 = (
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
            .where((tags_sum_3 >= min_total_hmer_lengths_in_tags) & (tags_sum_3 <= max_total_hmer_lengths_in_tags))
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

    # save to parquet
    if output_filename is not None:
        df_trimmer_histogram.to_parquet(output_filename)

    return df_trimmer_histogram
