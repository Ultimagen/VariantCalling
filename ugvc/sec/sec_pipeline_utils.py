import os
from typing import List, Dict

import pandas as pd
from simppl.simple_pipeline import SimplePipeline


def extract_relevant_gvcfs(sample_ids: List[str],
                           gvcf_files: List[str],
                           out_dir: str,
                           relevant_coords_file: str,
                           sp: SimplePipeline,
                           processes: int) -> List[str]:
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
                f'bcftools concat {" ".join(vcf_per_chr_files)} --rm-dups both -a'
                f" -Oz -o {relevant_gvcf}"
            )
        index_vcf_commands.append(f"tabix -p vcf {relevant_gvcf}")

        # extract variants in relevant coords (often from cloud to local storage)
        sp.run_parallel(extract_variants_commands, max_num_of_processes=processes)
        sp.run_parallel(index_chr_vcfs_commands, max_num_of_processes=processes)

        # concat vcf per chromosome and remove duplicate vcf records with disrupts indexing
        # (is this because of overlap in relevant coords?)
        try:
            sp.run_parallel(concat_vcf_commands, max_num_of_processes=processes)
            sp.run_parallel(index_vcf_commands, max_num_of_processes=processes)
        except RuntimeError:
            pass

        return gvcf_outputs


def read_sec_pipelines_inputs_table(args):
    inputs_table = pd.read_csv(args.inputs_table, sep="\t")
    sample_counts = inputs_table["sample_id"].value_counts()
    duplicated_samples = sample_counts[sample_counts > 1]
    if duplicated_samples.shape[0] > 0:
        raise RuntimeError(
            f"Found duplicated sample_ds in input:\n{duplicated_samples}"
        )
    return inputs_table