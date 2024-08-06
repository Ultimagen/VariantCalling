import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import H5Keys, Inputs, OutputFiles
from ugvc.utils.metrics_utils import merge_trimmer_histograms, read_sorter_statistics_csv, read_trimmer_failure_codes


def collect_statistics(input_files: Inputs, output_path: str, sample_name: str) -> Path:
    """
    Collect statistics from input files, parse and save them into h5 file

    Parameters
    ----------
    input_files : Inputs
        Input files containing the necessary data for statistics collection.
    output_path : str
        Path to the output directory.
    sample_name : str
        Sample name to be included as a prefix in the output files.

    Returns
    -------
    Path
        Path to the h5 file with statistics.
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
        store.put(H5Keys.STAR_STATS.value, star_stats)
        store.put(H5Keys.STAR_READS_PER_GENE.value, star_reads_per_gene, format="table")
        store.put(H5Keys.INSERT_QUALITY.value, insert_quality, format="table")
        store.put(
            H5Keys.INSERT_LENGTHS.value, pd.Series(insert_lengths), format="table"
        )

    return output_filename


def read_star_stats(star_stats_file: str) -> pd.Series:
    """
    Read STAR stats file (Log.final.out) and return parsed pandas object

    Parameters
    ----------
    star_stats_file : str
        Path to the STAR stats file.

    Returns
    -------
    pd.Series
        Series with parsed STAR stats.
    """
    df = pd.read_csv(star_stats_file, header=None, sep="\t")
    df.columns = ["metric", "value"]

    # parse metric description
    df["metric"] = df["metric"].str.replace("|", "").str.strip().str.replace(" ", "_")
    df.loc[:, "metric"] = df["metric"].str.replace(",", "").str.replace(":", "")

    # Add type (general, unique_reads, multi_mapping_reads, unmapped_reads, chimeric_reads)
    df.loc[:, "type"] = (
        df["metric"]
        .where(df["value"].isnull())
        .ffill()
        .fillna("general")
        .str.lower()
        .str.replace(":", "")
    )
    df = df.dropna(subset=["value"])

    # Add "pct_" to metric name if value ends with "%" (and remove "%" from the name)
    df.loc[df["value"].str.endswith("%"), "metric"] = df.loc[df["value"].str.endswith("%"), "metric"].apply(
        lambda x: "pct_" + x.replace("_%", "").replace("%_", "")
    )

    # Remove "%" from value
    df["value"] = df["value"].str.replace("%", "")

    # Set index
    df.set_index(['type', 'metric'], inplace=True)
    # convert df to pd.series for easier access
    s = df['value']
    return s


def get_insert_properties(insert, max_reads=None) -> tuple[pd.DataFrame, list[int]]:
    """
    Read insert subsample fastq.gz file and return quality scores per position and read lengths.

    Parameters
    ----------
    insert : str
        Path to the insert .fastq.gz file.
    max_reads : int, optional
        Maximum number of reads to process, by default None.

    Returns
    -------
    tuple[pd.DataFrame, list[int]]
        DataFrame with quality scores per position and list with read lengths.
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

    Parameters
    ----------
    h5_file : Path
        Path to the h5 file with statistics.
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
        mean_read_length = int(store[H5Keys.STAR_STATS.value].loc[('general','Average_input_read_length')]) + 1
        stats["mean_read_length"] = mean_read_length

        # %q >= 20 for insert
        q20 = store[H5Keys.SORTER_STATS.value].loc["% PF_Q20_bases"].value
        stats["PCT_q20"] = q20

        # %q >= 30 for insert
        q30 = store[H5Keys.SORTER_STATS.value].loc["% PF_Q30_bases"].value
        stats["PCT_q30"] = q30

        # %Aligned to genome
        ur_tmm = float(store[H5Keys.STAR_STATS.value].loc[('unmapped_reads','pct_of_reads_unmapped_too_many_mismatches')])
        ur_ts = float(store[H5Keys.STAR_STATS.value].loc[('unmapped_reads','pct_of_reads_unmapped_too_short')])
        ur_other = float(store[H5Keys.STAR_STATS.value].loc[('unmapped_reads','pct_of_reads_unmapped_other')])
        pct_aligned_to_genome = 100 - ur_tmm - ur_ts - ur_other
        stats["PCT_aligned_to_genome"] = pct_aligned_to_genome

        # %Assigned to genes (unique)
        unassigned_genes_df = store[H5Keys.STAR_READS_PER_GENE.value][
            store[H5Keys.STAR_READS_PER_GENE.value][0].astype(str).str.startswith("N_")
        ]  # unmapped, multimapping, noFeature, ambiguous
        unassigned_genes_unstranded = unassigned_genes_df.iloc[:, 1].sum()
        star_input_reads = int(store[H5Keys.STAR_STATS.value].loc[('general','Number_of_input_reads')])

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
        average_mapped_length = store[H5Keys.STAR_STATS.value].loc[('unique_reads','Average_mapped_length')]
        stats["average_mapped_length"] = average_mapped_length

        # Uniquely_mapped_reads_%
        pct_uniquely_mapped_reads =  store[H5Keys.STAR_STATS.value].loc[('unique_reads','pct_Uniquely_mapped_reads')]
        stats["PCT_uniquely_mapped_reads"] = pct_uniquely_mapped_reads

        # Mismatch_rate_per_base_%
        mismatch_rate = float(store[H5Keys.STAR_STATS.value].loc[('unique_reads','pct_Mismatch_rate_per_base')])
        stats["PCT_mismatch"] = mismatch_rate

        # PCT_deletion
        deletion_rate = float(store[H5Keys.STAR_STATS.value].loc[('unique_reads','pct_Deletion_rate_per_base')])
        stats["PCT_deletion"] = deletion_rate

        # PCT_insertion
        insertion_rate = float(store[H5Keys.STAR_STATS.value].loc[('unique_reads','pct_Insertion_rate_per_base')])
        stats["PCT_insertion"] = insertion_rate

    series = pd.Series(stats, dtype="float")
    series.to_hdf(h5_file, key=H5Keys.STATISTICS_SHORTLIST.value)
