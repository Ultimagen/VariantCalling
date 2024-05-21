import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import H5Keys, Inputs, OutputFiles
from ugvc.utils.metrics_utils import (
    merge_trimmer_histograms,
    read_sorter_statistics_csv,
    read_trimmer_failure_codes,
)


def collect_statistics(input_files: Inputs, output_path: str, sample_name: str) -> Path:
    """
    Collect statistics from input files, parse and save them into h5 file

    :param input_files: Inputs
    :param output_path: path to output directory
    :param sample_name: sample name to be include as a prefix in the output files

    :return: path to h5 file with statistics
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Merge Trimmer histograms from two optical paths (if needed)
    merged_histogram_csv = merge_trimmer_histograms(
        input_files.trimmer_histogram_csv, output_path=output_path
    )

    # Read inputs into df
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

    # Get insert subsample quality and lengths
    insert_quality, insert_lengths = get_insert_properties(input_files.insert)

    # Save statistics into h5
    output_filename = Path(output_path) / (sample_name + OutputFiles.H5.value)

    with pd.HDFStore(output_filename, "w") as store:
        store.put(H5Keys.TRIMMER_STATS.value, trimmer_stats, format="table")
        store.put(
            H5Keys.TRIMMER_FAILURE_CODES.value, df_trimmer_failure_codes, format="table"
        )
        store.put(H5Keys.TRIMMER_HISTOGRAM.value, histogram, format="table")
        store.put(H5Keys.SORTER_STATS.value, sorter_stats, format="table")
        store.put(H5Keys.STAR_STATS.value, star_stats, format="table")
        store.put(H5Keys.STAR_READS_PER_GENE.value, star_reads_per_gene, format="table")
        store.put(H5Keys.INSERT_QUALITY.value, insert_quality, format="table")
        store.put(
            H5Keys.INSERT_LENGTHS.value, pd.Series(insert_lengths), format="table"
        )

    return output_filename


def read_star_stats(star_stats_file: str) -> pd.DataFrame:
    """
    Read STAR stats file (Log.final.out) and return parsed DataFrame

    :param star_stats_file: path to STAR stats file

    :return: DataFrame with parsed STAR stats
    """
    df = pd.read_csv(star_stats_file, header=None, sep="\t")
    df.columns = ["metric", "value"]

    # parse metric description
    df["metric"] = df["metric"].str.replace("|", "").str.strip().str.replace(" ", "_")
    df.loc[:, "metric"] = df["metric"].str.replace(",", "").str.replace(":", "")

    # Add read_type (general, unique_reads, multi_mapping_reads, unmapped_reads, chimeric_reads)
    df.loc[:, "read_type"] = (
        df["metric"]
        .where(df["value"].isnull())
        .ffill()
        .fillna("general")
        .str.lower()
        .str.replace(":", "")
    )
    df = df.dropna(subset=["value"])

    # parse value: add value type (number, percentage, datetime)
    df["value_type"] = "number"
    df.loc[df["value"].str.endswith("%"), "value_type"] = "percentage"
    df.loc[df["value"].str.find(":") != -1, "value_type"] = "datetime"
    df["value"] = df["value"].str.replace("%", "")
    return df


def get_insert_properties(insert, max_reads=None) -> tuple[pd.DataFrame, list[int]]:
    """
    Read insert subsample fastq.gz file and return quality scores per position and read lengths.

    :param insert: path to insert .fastq.gz file
    :param max_reads: maximum number of reads to process

    :return: DataFrame with quality scores per position and list with read lengths
    """
    insert_lengths = []
    counter = defaultdict(lambda: defaultdict(int))

    # read fastq file
    fastq_parser = SeqIO.parse(gzip.open(insert, "rt"), "fastq")

    for j, record in enumerate(fastq_parser):
        insert_lengths.append(len(record))
        score = record.letter_annotations["phred_quality"]
        for i in range(len(score)):
            counter[i + 1][score[i]] += 1
        if max_reads and j > max_reads:
            break

    df_insert_quality = pd.DataFrame(counter).sort_index()
    df_insert_quality.index.name = "quality"
    df_insert_quality.columns.name = "position"

    # normalize
    df_insert_quality = df_insert_quality / df_insert_quality.sum().sum()

    return df_insert_quality, insert_lengths


def extract_statistics_table(h5_file: Path):
    """
    Create shortlist of statistics from h5 file and append it to h5 file.

    :param h5_file: path to h5 file with statistics
    """
    stats = {}

    with pd.HDFStore(h5_file, "r") as store:
        # number of Input Reads
        num_input_reads = store[H5Keys.TRIMMER_STATS.value]["num input reads"].values[0]
        stats["num_input_reads"] = num_input_reads

        # number of Trimmed reads
        num_trimmed_reads = store[H5Keys.TRIMMER_STATS.value][
            "num trimmed reads"
        ].values[0]
        stats["num_trimmed_reads"] = num_trimmed_reads

        # PCT_pass_trimmer
        pass_trimmer_rate = num_trimmed_reads / num_input_reads
        stats["PCT_pass_trimmer"] = pass_trimmer_rate * 100

        # Mean UMI per cell
        mean_umi_per_cell = None  # TODO: waiting for the calculation details from Gila
        stats["mean_umi_per_cell"] = mean_umi_per_cell

        # Mean read length
        mean_read_length = (
            int(
                store[H5Keys.STAR_STATS.value][
                    store[H5Keys.STAR_STATS.value]["metric"]
                    == "Average_input_read_length"
                ]["value"].values[0]
            )
            + 1
        )
        stats["mean_read_length"] = mean_read_length

        # %q >= 20 for insert
        q20 = store[H5Keys.SORTER_STATS.value].loc["% PF_Q20_bases"].value
        stats["PCT_q20"] = q20

        # %q >= 30 for insert
        q30 = store[H5Keys.SORTER_STATS.value].loc["% PF_Q30_bases"].value
        stats["PCT_q30"] = q30

        # %Aligned to genome
        unmapped_reads_df = store[H5Keys.STAR_STATS.value].loc[
            (store[H5Keys.STAR_STATS.value]["read_type"] == "unmapped_reads")
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
        pct_aligned_to_genome = 100 - ur_tmm - ur_ts - ur_other
        stats["PCT_aligned_to_genome"] = pct_aligned_to_genome

        # %Assigned to genes (unique)
        unassigned_genes_df = store[H5Keys.STAR_READS_PER_GENE.value][
            store[H5Keys.STAR_READS_PER_GENE.value][0].astype(str).str.startswith("N_")
        ]  # unmapped, multimapping, noFeature, ambiguous
        unassigned_genes_unstranded = unassigned_genes_df.iloc[:, 1].sum()
        star_input_reads = int(
            store[H5Keys.STAR_STATS.value]
            .loc[
                (store[H5Keys.STAR_STATS.value]["read_type"] == "general")
                & (store[H5Keys.STAR_STATS.value]["metric"] == "Number_of_input_reads"),
                "value",
            ]
            .values[0]
        )
        pct_aligned_to_genes_unstranded = (
            100 * (star_input_reads - unassigned_genes_unstranded) / star_input_reads
        )
        stats["PCT_aligned_to_genes_unstranded"] = pct_aligned_to_genes_unstranded

        # %Assigned to genes (unique; forward)
        unassigned_genes_forward = unassigned_genes_df.iloc[:, 2].sum()
        pct_aligned_to_genes_forward = (
            100 * (star_input_reads - unassigned_genes_forward) / star_input_reads
        )
        stats["PCT_aligned_to_genes_forward"] = pct_aligned_to_genes_forward

        # %Assigned to genes (unique; reverse)
        unassigned_genes_reverse = unassigned_genes_df.iloc[:, 3].sum()
        pct_aligned_to_genes_reverse = (
            100 * (star_input_reads - unassigned_genes_reverse) / star_input_reads
        )
        stats["PCT_aligned_to_genes_reverse"] = pct_aligned_to_genes_reverse

        # Average_mapped_length
        unique_reads_df = store[H5Keys.STAR_STATS.value].loc[
            (store[H5Keys.STAR_STATS.value]["read_type"] == "unique_reads")
        ]
        average_mapped_length = unique_reads_df.loc[
            unique_reads_df["metric"] == "Average_mapped_length", "value"
        ].values[0]
        stats["average_mapped_length"] = average_mapped_length

        # Uniquely_mapped_reads_%
        pct_uniquely_mapped_reads = unique_reads_df.loc[
            unique_reads_df["metric"] == "Uniquely_mapped_reads_%", "value"
        ].values[0]
        stats["PCT_uniquely_mapped_reads"] = pct_uniquely_mapped_reads

        # Mismatch_rate_per_base_%
        mismatch_rate = float(
            unique_reads_df.loc[
                unique_reads_df["metric"] == "Mismatch_rate_per_base_%", "value"
            ].values[0]
        )
        stats["PCT_mismatch"] = mismatch_rate

        # PCT_deletion
        deletion_rate = float(
            unique_reads_df.loc[
                unique_reads_df["metric"] == "Deletion_rate_per_base", "value"
            ].values[0]
        )
        stats["PCT_deletion"] = deletion_rate

        # PCT_insertion
        insertion_rate = float(
            unique_reads_df.loc[
                unique_reads_df["metric"] == "Insertion_rate_per_base", "value"
            ].values[0]
        )
        stats["PCT_insertion"] = insertion_rate

    df = pd.DataFrame(stats.items(), columns=["statistic", "value"])
    df.set_index("statistic", inplace=True)
    df["value"] = df["value"].astype("float")
    with pd.HDFStore(h5_file, "a") as store:
        store.put(H5Keys.STATISTICS_SHORTLIST.value, df, format="table")
