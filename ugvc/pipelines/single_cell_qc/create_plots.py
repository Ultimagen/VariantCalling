from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import H5Keys, OutputFiles
from ugvc.utils.misc_utils import set_pyplot_defaults

set_pyplot_defaults()


def cbc_umi_plot(h5_file: str, output_path: str) -> Path:
    """
    Count number of unique UMI per CBC, to get a rough estimate of the number of cells in the sample.

    Parameters
    ----------
    h5_file : str
        Path to h5 file with statistics.
    output_path : str
        Path to output directory.

    Returns
    -------
    Path
        Path to the plot.
    """
    with pd.HDFStore(h5_file, "r") as store:
        histogram = store[H5Keys.TRIMMER_HISTOGRAM.value]

    umi_col = histogram.columns[histogram.columns.str.contains("UMI")][0]
    cbc_columns = list(set(histogram) - set([umi_col, "count"]))

    # Counting how many distinct UMIs there are per cell barcode
    cbc_num_umi_df = (
        histogram.drop(columns=[umi_col, "count"]).groupby(cbc_columns).size().reset_index(name="Num Unique UMI")
    )

    # Sorting by Num UMI and setting a column that will be the CBC index
    plot_df = (
        cbc_num_umi_df.sort_values("Num Unique UMI", ascending=False)
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "CBC"})
    )

    # Plotting
    plt.figure()
    ax = plt.gca()

    ax = sns.scatterplot(data=plot_df, x="CBC", y="Num Unique UMI", linewidth=0)
    ax.set(yscale="log", xscale="log", title="Barcode Rank")

    plot_file = Path(output_path) / OutputFiles.CBC_UMI_PLOT.value
    plt.savefig(plot_file)
    plt.close()
    return plot_file


def plot_insert_length_histogram(h5_file: str, output_path: str) -> Path:
    """
    Plot histogram of insert lengths.

    Parameters
    ----------
    h5_file : str
        Path to h5 file with statistics.
    output_path : str
        Path to output directory.

    Returns
    -------
    Path
        Path to the plot.
    """
    with pd.HDFStore(h5_file, "r") as store:
        insert_lengths = store[H5Keys.INSERT_LENGTHS.value]

    # Calculate IQR
    Q1 = np.percentile(insert_lengths, 25)
    Q3 = np.percentile(insert_lengths, 75)
    IQR = Q3 - Q1

    # Calculate bin width using Freedman-Diaconis rule
    bin_width = 2 * IQR * len(insert_lengths) ** (-1 / 3)
    if bin_width == 0:  # if all values are the same or if the data is extremely skewed the bin width will be 0
        bins = 10  # Default value to avoid division by zero
    else:
        bins = int((max(insert_lengths) - min(insert_lengths)) / bin_width)

    pd.Series(insert_lengths).hist(bins=bins, density=True)

    plt.xlabel("Read Length")
    plt.ylabel("Frequency")
    plt.title("Insert Length Histogram")

    plot_file = Path(output_path) / OutputFiles.INSERT_LENGTH_HISTOGRAM.value
    plt.savefig(plot_file)
    plt.close()
    return plot_file


def plot_mean_insert_quality_histogram(h5_file: str, output_path: str) -> Path:
    """
    Plot histogram of mean insert quality.

    Parameters
    ----------
    h5_file : str
        Path to h5 file with statistics.
    output_path : str
        Path to output directory.

    Returns
    -------
    Path
        Path to the plot.
    """
    with pd.HDFStore(h5_file, "r") as store:
        insert_quality = store[H5Keys.INSERT_QUALITY.value]

    # histogram of overall quality
    qual_hist = insert_quality.sum(axis=1)
    qual_hist.plot.bar()
    plt.xticks(rotation=0)
    plt.xlabel("Quality")
    plt.ylabel("Frequency")
    plt.title("Mean Insert Quality Histogram")

    plot_file = Path(output_path) / OutputFiles.MEAN_INSERT_QUALITY_PLOT.value
    plt.savefig(plot_file)
    plt.close()
    return plot_file


def plot_quality_per_position(h5_file: str, output_path: str) -> Path:
    """
    Plot quality per position for the insert, with percentiles.

    Parameters
    ----------
    h5_file : str
        Path to h5 file with statistics.
    output_path : str
        Path to output directory.

    Returns
    -------
    Path
        Path to the plot.
    """
    with pd.HDFStore(h5_file, "r") as store:
        insert_quality = store[H5Keys.INSERT_QUALITY.value]

    # quality percentiles per position
    df_cdf = insert_quality.cumsum() / insert_quality.sum()
    percentiles = {q: (df_cdf >= q).idxmax() for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    plt.figure()
    plt.fill_between(
        percentiles[0.05].index,
        percentiles[0.05],
        percentiles[0.95],
        color="b",
        alpha=0.2,
        label="5-95%",
    )
    plt.fill_between(
        percentiles[0.25].index,
        percentiles[0.25],
        percentiles[0.75],
        color="b",
        alpha=0.5,
        label="25-75%",
    )
    plt.plot(percentiles[0.5].index, percentiles[0.5], color="k", label="median", linewidth=2)
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Quality")
    plt.title("Quality Per Position")

    plot_file = Path(output_path) / OutputFiles.QUALITY_PER_POSITION_PLOT.value
    plt.savefig(plot_file)
    plt.close()
    return plot_file
