import pandas as pd

def read_sorter_statistics_csv(sorter_stats_csv: str, edit_metric_names: bool = True) -> pd.Series:
    """
    Collect sorter statistics from csv

    Parameters
    ----------
    sorter_stats_csv : str
        path to a Sorter stats file
    edit_metric_names: bool
        if True, edit the metric names to be consistent in the naming of percentages

    Returns
    -------
    pd.Series
        Series with sorter statistics
    """

    # read Sorter stats
    df_sorter_stats = pd.read_csv(sorter_stats_csv, header=None, names=["metric", "value"]).set_index("metric")
    # replace '(' and ')' in values (legacy format for F95)
    df_sorter_stats = df_sorter_stats.assign(
        value=df_sorter_stats["value"]
        .astype(str)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .astype(float)
        .values
    )
    # add convenient metric
    if "Failed_QC_reads" in df_sorter_stats.index and "PF_Barcode_reads" in df_sorter_stats.index:
        df_sorter_stats.loc["PCT_Failed_QC_reads"] = (
            100
            * df_sorter_stats.loc["Failed_QC_reads"]
            / (df_sorter_stats.loc["Failed_QC_reads"] + df_sorter_stats.loc["PF_Barcode_reads"])
        )

    if edit_metric_names:
        # rename metrics to uniform convention
        df_sorter_stats = df_sorter_stats.rename({c: c.replace("% ", "PCT_") for c in df_sorter_stats.index})
    return df_sorter_stats["value"]
