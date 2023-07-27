import pandas as pd

from ugvc.utils.misc_utils import set_pyplot_defaults

set_pyplot_defaults()


# Trimmer segment labels
T_HMER_START = "T_hmer_start"
T_HMER_END = "T_hmer_end"
A_HMER_START = "A_hmer_start"
A_HMER_END = "A_hmer_end"
NATIVE_ADAPTER = "native_adapter_with_leading_C"
STEM_END = "Stem_end"  # when native adapter trimming was done on-tool a modified format is used
# Category names
END_UNREACHED = "END_UNREACHED"
HYB = "HYB"
LIG = "LIG"
MIXED = "MIXED"
UNDETERMINED = "UNDETERMINED"
# Internal names
COUNT = "count"
STRAND_RATIO_START = "strand_ratio_start"
STRAND_RATIO_END = "strand_ratio_end"
STRAND_RATIO_CATEGORY_START = "strand_ratio_category_start"
STRAND_RATIO_CATEGORY_END = "strand_ratio_category_end"
# Input parameters
STRAND_RATIO_LOWER_THRESH = 0.27
STRAND_RATIO_UPPER_THRESH = 0.73
MIN_TOTAL_HMER_LENGTHS_IN_TAGS = 4
MAX_TOTAL_HMER_LENGTHS_IN_TAGS = 8


def get_annotation(x, sr_lower, sr_upper):
    if x == 0:
        return HYB
    if x == 1:
        return LIG
    if sr_lower <= x <= sr_upper:
        return MIXED
    return UNDETERMINED


def read_balanced_epcr_trimmer_histogram(
    trimmer_histogram,
    sr_lower=STRAND_RATIO_LOWER_THRESH,
    sr_upper=STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags=MIN_TOTAL_HMER_LENGTHS_IN_TAGS,
    max_total_hmer_lengths_in_tags=MAX_TOTAL_HMER_LENGTHS_IN_TAGS,
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
        If required columns are missing ({COUNT}, {T_HMER_START}, {A_HMER_START})
    """
    # read histogram
    df_trimmer_histogram = pd.read_csv(trimmer_histogram)
    # change legacy segment names
    df_trimmer_histogram = df_trimmer_histogram.rename(
        columns={
            "T hmer": T_HMER_START,
            "A hmer": A_HMER_START,
            "A_hmer_5": A_HMER_START,
            "T_hmer_5": T_HMER_START,
            "A_hmer_3": A_HMER_END,
            "T_hmer_3": T_HMER_END,
        }
    )
    # make sure expected columns exist
    for col in (COUNT, T_HMER_START, A_HMER_START):
        if col not in df_trimmer_histogram.columns:
            raise ValueError(f"Missing expected column {col} in {trimmer_histogram}")
    if (A_HMER_END in df_trimmer_histogram.columns or T_HMER_END in df_trimmer_histogram.columns) and (
        NATIVE_ADAPTER not in df_trimmer_histogram.columns and STEM_END not in df_trimmer_histogram.columns
    ):
        # If an end tag exists (LA-v6)
        raise ValueError(f"Missing expected column {NATIVE_ADAPTER} or {STEM_END} in {trimmer_histogram}")

    df_trimmer_histogram.index.name = sample_name

    # add normalized count column
    df_trimmer_histogram = df_trimmer_histogram.assign(
        count_norm=df_trimmer_histogram[COUNT] / df_trimmer_histogram[COUNT].sum()
    )
    # add strand ratio columns and determine categories
    tags_sum_5 = df_trimmer_histogram[T_HMER_START] + df_trimmer_histogram[A_HMER_START]
    df_trimmer_histogram.loc[:, STRAND_RATIO_START] = (
        (df_trimmer_histogram[T_HMER_START] / (tags_sum_5))
        .where((tags_sum_5 >= min_total_hmer_lengths_in_tags) & (tags_sum_5 <= max_total_hmer_lengths_in_tags))
        .round(2)
    )
    # determine strand ratio category
    df_trimmer_histogram.loc[:, STRAND_RATIO_CATEGORY_START] = df_trimmer_histogram[STRAND_RATIO_START].apply(
        lambda x: get_annotation(x, sr_lower, sr_upper)
    )
    if A_HMER_END in df_trimmer_histogram.columns or T_HMER_END in df_trimmer_histogram.columns:
        # if only one of the end tags exists (maybe a small subsample) assign the other to 0
        for c in (A_HMER_END, T_HMER_END):
            if c not in df_trimmer_histogram.columns:
                df_trimmer_histogram.loc[:, c] = 0

        tags_sum_3 = df_trimmer_histogram[T_HMER_END] + df_trimmer_histogram[A_HMER_END]
        df_trimmer_histogram.loc[:, STRAND_RATIO_END] = (
            (df_trimmer_histogram[T_HMER_END] / (df_trimmer_histogram[T_HMER_END] + df_trimmer_histogram[A_HMER_END]))
            .where((tags_sum_3 >= min_total_hmer_lengths_in_tags) & (tags_sum_3 <= max_total_hmer_lengths_in_tags))
            .round(2)
        )
        # determine if end was reached - at least 1bp native adapter or all of the end stem were found
        is_end_reached = (
            df_trimmer_histogram[NATIVE_ADAPTER] >= 1
            if NATIVE_ADAPTER in df_trimmer_histogram.columns
            else df_trimmer_histogram[STEM_END] >= 11
        )
        # determine strand ratio category
        df_trimmer_histogram.loc[:, STRAND_RATIO_CATEGORY_END] = (
            df_trimmer_histogram[STRAND_RATIO_END]
            .apply(lambda x: get_annotation(x, sr_lower, sr_upper))
            .where(is_end_reached, END_UNREACHED)
        )

    # save to parquet
    if output_filename is not None:
        df_trimmer_histogram.to_parquet(output_filename)

    return df_trimmer_histogram
