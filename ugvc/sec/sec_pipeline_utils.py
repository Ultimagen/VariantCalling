from __future__ import annotations

import os

import pandas as pd
from pandas import DataFrame
from simppl.simple_pipeline import SimplePipeline


def extract_relevant_gvcfs(
    sample_ids: list[str],
    gvcf_files: list[str],
    out_dir: str,
    relevant_coords_file: str,
    simple_pipeline: SimplePipeline,
    processes: int,
) -> list[str]:
    """
    Intersect the remote gvcf files with relevant_coords_file, and save result in local storage

    Parameters
    ----------
    sample_ids: list[str]
        the ids of sample, ordered like gvcf files
    gvcf_files: list[str]
        urls of gvcf files on GCS
    out_dir: str
        output directory
    relevant_coords_file: str
        bed file describing relevanr coordinates for analysis
    simple_pipeline: SimplePipeline
        object which will run the extraction commands
    processes: int
        The number of parallel processes to use

    Returns
    -------
    list[str]:
        List of extracted gvcf files
    """
    extract_variants_commands = []
    index_chr_vcfs_commands = []
    concat_vcf_commands = []
    index_vcf_commands = []
    authorize_gcp_command = "export GCS_OAUTH_TOKEN=`gcloud auth application-default print-access-token`"
    relevant_chromosomes = list(pd.read_csv(relevant_coords_file, sep="\t", header=None)[0].unique())
    os.makedirs(f"{out_dir}/gvcf", exist_ok=True)
    gvcf_outputs = []

    for sample_id, gvcf_file in zip(sample_ids, gvcf_files):
        relevant_gvcf = f"{out_dir}/gvcf/{sample_id}.g.vcf.gz"
        gvcf_outputs.append(relevant_gvcf)
        vcf_per_chr_files = []
        for chromosome in relevant_chromosomes:
            vcf_file_per_chr = f"{relevant_gvcf}.{chromosome}.vcf.gz"
            vcf_per_chr_files.append(vcf_file_per_chr)

            if not os.path.exists(relevant_gvcf):
                extract_variants_commands.append(
                    f"{authorize_gcp_command}; "
                    f"bcftools view {gvcf_file} {chromosome} -Oz"
                    f" | bedtools intersect -a stdin -b {relevant_coords_file} -header"
                    f" | uniq | bgzip > {vcf_file_per_chr}"
                )
                index_chr_vcfs_commands.append(f"tabix -p vcf {vcf_file_per_chr}")
        if not os.path.exists(relevant_gvcf):
            concat_vcf_commands.append(
                f'bcftools concat {" ".join(vcf_per_chr_files)} --rm-dups both -a' f" -Oz -o {relevant_gvcf}"
            )
        index_vcf_commands.append(f"tabix -p vcf {relevant_gvcf}")

        # extract variants in relevant coords (often from cloud to local storage)
        simple_pipeline.run_parallel(extract_variants_commands, max_num_of_processes=processes)
        simple_pipeline.run_parallel(index_chr_vcfs_commands, max_num_of_processes=processes)

        # concat vcf per chromosome and remove duplicate vcf records with disrupts indexing
        # (is this because of overlap in relevant coords?)
        try:
            simple_pipeline.run_parallel(concat_vcf_commands, max_num_of_processes=processes)
            simple_pipeline.run_parallel(index_vcf_commands, max_num_of_processes=processes)
        except RuntimeError:
            pass

    return gvcf_outputs


def read_sec_pipelines_inputs_table(inputs_table_file: str) -> DataFrame:
    """

    Parameters
    ----------
    inputs_table_file: str
        path of tsv file specifying SEC inputs [Workflow ID, sample_id, gvcf]

    Returns
    -------
    Dataframe:
         Same schema, after validation

    Raises
    ------
    RuntimeError
        When duplicated sample is found in input

    """
    inputs_table = pd.read_csv(inputs_table_file, sep="\t")
    sample_counts = inputs_table["sample_id"].value_counts()
    duplicated_samples = sample_counts[sample_counts > 1]
    if duplicated_samples.shape[0] > 0:
        raise RuntimeError(f"Found duplicated sample_ds in input:\n{duplicated_samples}")
    return inputs_table
