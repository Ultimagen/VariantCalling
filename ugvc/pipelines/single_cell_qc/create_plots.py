import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ugvc.utils.misc_utils import set_pyplot_defaults

set_pyplot_defaults()


def cbc_umi_plot(h5_file: str, output_path: str):
    "Count number of unique UMI per CBC, to get a rough estimate of the number of cells in the sample"
    with pd.HDFStore(h5_file, "r") as store:
        histogram = store["trimmer_histogram"]

    umi_col = histogram.columns[histogram.columns.str.contains("UMI")][0]
    cbc_columns = list(set(histogram) - set([umi_col, "count"]))

    # Counting how many distinct UMIs there are per cell barcode
    cbc_num_umi_df = (
        histogram.drop(columns=[umi_col, "count"])
        .groupby(cbc_columns)
        .size()
        .reset_index(name="Num Unique UMI")
    )

    # Sorting by Num UMI and setting a column that will be the CBC index
    plot_df = (
        cbc_num_umi_df.sort_values("Num Unique UMI", ascending=False)
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "CBC"})
    )

    # Plotting
    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    ax = sns.scatterplot(data=plot_df, x="CBC", y="Num Unique UMI", linewidth=0)
    ax.set(yscale="log", xscale="log", title="Barcode Rank")

    plot_file = os.path.join(output_path, "cbc_umi_plot.png")
    plt.savefig(plot_file)
    plt.close()
    return plot_file


def plot_r2_length_histogram(h5_file: str, output_path: str) -> str:
    with pd.HDFStore(h5_file, "r") as store:
        r2_lengths = store["r2_lengths"]
    pd.Series(r2_lengths).hist(color="gray", bins=1000)

    plt.xlabel("Read Length")
    plt.ylabel("Frequency")
    plt.title("Insert Length Histogram")
    plot_file = os.path.join(output_path, "r2_length_histogram.png")
    plt.savefig(plot_file)
    plt.close()
    return plot_file


def plot_mean_r2_quality_histogram(h5_file: str, output_path: str) -> str:
    with pd.HDFStore(h5_file, "r") as store:
        r2_quality = store["r2_quality"]

    # histogram of overall quality
    qual_hist = r2_quality.sum(axis=1)
    qual_hist.plot()
    plt.xlabel("Quality")
    plt.ylabel("Frequency")
    plt.title("Mean insert Quality Histogram")

    plot_file = os.path.join(output_path, "mean_r2_quality_histogram.png")
    plt.savefig(plot_file)
    plt.close()
    return plot_file


def plot_quality_per_position(h5_file: str, output_path: str) -> str:
    with pd.HDFStore(h5_file, "r") as store:
        r2_quality = store["r2_quality"]

    # quality percentiles per position
    df_cdf = r2_quality.cumsum() / r2_quality.sum()
    percentiles = {q: (df_cdf >= q).idxmax() for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
    plt.figure(figsize=(10, 5))
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
    plt.plot(
        percentiles[0.5].index, percentiles[0.5], color="k", label="median", linewidth=2
    )
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Quality")
    plt.title("Quality per Position")

    plot_file = os.path.join(output_path, "quality_per_position.png")
    plt.savefig(plot_file)
    plt.close()
    return plot_file
