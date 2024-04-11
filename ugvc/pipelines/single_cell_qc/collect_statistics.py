import gzip
import os
from collections import defaultdict

import pandas as pd
from Bio import SeqIO

from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import Inputs
from ugvc.utils.metrics_utils import (
    merge_trimmer_histograms,
    read_sorter_statistics_csv,
    read_trimmer_failure_codes,
)


def collect_statistics(input_files: Inputs, output_path: str) -> str:
    os.makedirs(output_path, exist_ok=True)

    # Merge Trimmer histograms from two optical paths
    merged_histogram_csv = merge_trimmer_histograms(
        input_files.trimmer_histogram_csv, output_path=output_path
    )

    # Read inputs into df_r2_quality
    trimmer_stats = pd.read_csv(input_files.trimmer_stats_csv)
    df_trimmer_failure_codes = read_trimmer_failure_codes(
        input_files.trimmer_failure_codes_csv,
        add_total=True,
    )
    histogram = pd.read_csv(merged_histogram_csv)
    sorter_stats = read_sorter_statistics_csv(input_files.sorter_stats_csv)
    star_stats = read_star_stats(input_files.star_stats)
    star_reads_per_gene = pd.read_csv(
        input_files.star_reads_per_gene, header=None, sep="\t"
    )

    # Get R2 subsample quality and lengths
    r2_quality, r2_lengths = get_r2_properties(input_files.r2_subsample)

    # Save statistics into h5
    output_filename = os.path.join(output_path, "single_cell_qc_stats.h5")

    with pd.HDFStore(output_filename, "w") as store:
        store["trimmer_stats"] = trimmer_stats
        store["trimmer_failure_codes"] = df_trimmer_failure_codes
        store["trimmer_histogram"] = histogram
        store["sorter_stats"] = sorter_stats
        store["star_stats"] = star_stats
        store["star_reads_per_gene"] = star_reads_per_gene
        store["r2_quality"] = r2_quality
        store["r2_lengths"] = pd.Series(r2_lengths)

    return output_filename


def read_star_stats(star_stats_file: str) -> pd.DataFrame:
    df_r2_quality = pd.read_csv(star_stats_file, header=None, sep="\t")
    df_r2_quality.columns = ["metric", "value"]

    # parse metric description
    df_r2_quality["metric"] = (
        df_r2_quality["metric"].str.replace("|", "").str.strip().str.replace(" ", "_")
    )
    df_r2_quality.loc[:, "metric"] = (
        df_r2_quality["metric"].str.replace(",", "").str.replace(":", "")
    )

    # Add read_type (general, unique_reads, multi_mapping_reads, unmapped_reads, chimeric_reads)
    df_r2_quality.loc[:, "read_type"] = (
        df_r2_quality["metric"]
        .where(df_r2_quality["value"].isnull())
        .ffill()
        .fillna("general")
        .str.lower()
        .str.replace(":", "")
    )
    df_r2_quality = df_r2_quality.dropna(subset=["value"])

    # parse value: add value type (number, percentage, datetime)
    df_r2_quality["value_type"] = "number"
    df_r2_quality.loc[df_r2_quality["value"].str.endswith("%"), "value_type"] = (
        "percentage"
    )
    df_r2_quality.loc[df_r2_quality["value"].str.find(":") != -1, "value_type"] = (
        "datetime"
    )
    df_r2_quality["value"] = df_r2_quality["value"].str.replace("%", "")
    return df_r2_quality


def get_r2_properties(r2_subsample, max_reads=None):
    """
    Read R2 subsample fastq.gz file and return quality scores per position and read lengths
    """
    r2_lengths = []
    counter = defaultdict(lambda: defaultdict(int))

    fastq_parser = SeqIO.parse(gzip.open(r2_subsample, "rt"), "fastq")

    for j, record in enumerate(fastq_parser):
        r2_lengths.append(len(record))
        score = record.letter_annotations["phred_quality"]
        for i in range(len(score)):
            counter[i + 1][score[i]] += 1
        if max_reads and j > max_reads:
            break

    df_r2_quality = pd.DataFrame(counter).sort_index()
    df_r2_quality.index.name = "quality"
    df_r2_quality.columns.name = "position"
    # normalize
    df_r2_quality = df_r2_quality / df_r2_quality.sum().sum()

    return df_r2_quality, r2_lengths


def extract_statistics_table(h5_file: str) -> str:
    stats = {}

    with pd.HDFStore(h5_file, "r") as store:
        # number of Input Reads
        num_input_reads = store["trimmer_stats"]["num input reads"].values[0]
        stats["num_input_reads"] = num_input_reads

        # number of Trimmed reads
        num_trimmed_reads = store["trimmer_stats"]["num trimmed reads"].values[0]
        stats["num_trimmed_reads"] = num_trimmed_reads

        # Pass_Trimmer_Rate
        pass_trimmer_rate = num_trimmed_reads / num_input_reads
        stats["pass_trimmer_rate"] = pass_trimmer_rate

        # Mean UMI per cell
        mean_umi_per_cell = 0  # TODO: calculate based on store["trimmer_histogram"]?
        stats["mean_umi_per_cell"] = mean_umi_per_cell

        # Mean read length
        mean_read_length = (
            int(
                store["star_stats"][
                    store["star_stats"]["metric"] == "Average_input_read_length"
                ]["value"].values[0]
            )
            + 1
        )
        stats["mean_read_length"] = mean_read_length

        # %q >= 20 for R2
        q20 = store["sorter_stats"].loc["% PF_Q20_bases"].value
        stats["%q20"] = q20

        # %q >= 30 for R2
        q30 = store["sorter_stats"].loc["% PF_Q30_bases"].value
        stats["%q30"] = q30

        # %Aligned to genome
        unmapped_reads_df = store["star_stats"].loc[
            (store["star_stats"]["read_type"] == "unmapped_reads")
        ]
        ur_tmm = float(
            unmapped_reads_df.loc[
                unmapped_reads_df["metric"]
                == "%_of_reads_unmapped_too_many_mismatches",
                "value",
            ].values[0]
        )
        ur_ts = float(
            unmapped_reads_df.loc[
                unmapped_reads_df["metric"] == "%_of_reads_unmapped_too_short", "value"
            ].values[0]
        )
        ur_other = float(
            unmapped_reads_df.loc[
                unmapped_reads_df["metric"] == "%_of_reads_unmapped_other", "value"
            ].values[0]
        )
        prc_aligned_to_genome = 100 - ur_tmm - ur_ts - ur_other
        stats["prc_aligned_to_genome"] = prc_aligned_to_genome

        # %Assigned to genes (unique)
        unassigned_genes_df = store["star_reads_per_gene"][
            store["star_reads_per_gene"][0].astype(str).str.startswith("N_")
        ]  # unmapped, multimapping, noFeature, ambiguous
        unassigned_genes_unstranded = unassigned_genes_df.iloc[:, 1].sum()
        star_input_reads = int(
            store["star_stats"]
            .loc[
                (store["star_stats"]["read_type"] == "general")
                & (store["star_stats"]["metric"] == "Number_of_input_reads"),
                "value",
            ]
            .values[0]
        )
        prc_aligned_to_genes_unstranded = (
            100 * (star_input_reads - unassigned_genes_unstranded) / star_input_reads
        )
        stats["prc_aligned_to_genes_unstranded"] = prc_aligned_to_genes_unstranded

        # %Assigned to genes (unique; forward)
        unassigned_genes_forward = unassigned_genes_df.iloc[:, 2].sum()
        prc_aligned_to_genes_forward = (
            100 * (star_input_reads - unassigned_genes_forward) / star_input_reads
        )
        stats["prc_aligned_to_genes_forward"] = prc_aligned_to_genes_forward

        # %Assigned to genes (unique; reverse)
        unassigned_genes_reverse = unassigned_genes_df.iloc[:, 3].sum()
        prc_aligned_to_genes_reverse = (
            100 * (star_input_reads - unassigned_genes_reverse) / star_input_reads
        )
        stats["prc_aligned_to_genes_reverse"] = prc_aligned_to_genes_reverse

        # Average_mapped_length
        unique_reads_df = store["star_stats"].loc[
            (store["star_stats"]["read_type"] == "unique_reads")
        ]
        average_mapped_length = unique_reads_df.loc[
            unique_reads_df["metric"] == "Average_mapped_length", "value"
        ].values[0]
        stats["average_mapped_length"] = average_mapped_length

        # Uniquely_mapped_reads_%
        prc_uniquely_mapped_reads = unique_reads_df.loc[
            unique_reads_df["metric"] == "Uniquely_mapped_reads_%", "value"
        ].values[0]
        stats["prc_uniquely_mapped_reads"] = prc_uniquely_mapped_reads

        # Mismatch_rate_per_base_%
        prc_mismatch_rate_per_base = unique_reads_df.loc[
            unique_reads_df["metric"] == "Mismatch_rate_per_base_%", "value"
        ].values[0]
        stats["prc_mismatch_rate_per_base"] = prc_mismatch_rate_per_base

        # Deletion_rate_per_base
        deletion_rate_per_base = unique_reads_df.loc[
            unique_reads_df["metric"] == "Deletion_rate_per_base", "value"
        ].values[0]
        stats["deletion_rate_per_base"] = deletion_rate_per_base

        # Insertion_rate_per_base
        insertion_rate_per_base = unique_reads_df.loc[
            unique_reads_df["metric"] == "Insertion_rate_per_base", "value"
        ].values[0]
        stats["insertion_rate_per_base"] = insertion_rate_per_base

    df = pd.DataFrame(stats.items(), columns=["statistic", "value"])
    df.set_index("statistic", inplace=True)
    df["value"] = df["value"].astype("float")
    with pd.HDFStore(h5_file, "a") as store:
        store["statistics_shortlist"] = df
