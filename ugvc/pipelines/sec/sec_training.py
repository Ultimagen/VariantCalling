#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Full SEC-training pipeline
# CHANGELOG in reverse chronological order

import os
import sys

from simppl.cli import get_parser
from simppl.simple_pipeline import SimplePipeline

from ugvc import base_dir as ugvc_pkg
from ugvc.sec.sec_pipeline_utils import extract_relevant_gvcfs, read_sec_pipelines_inputs_table
from ugvc.utils.consts import FileExtension


def parse_args(argv):
    parser = get_parser("sec_training", run.__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--inputs_table",
        required=True,
        help="A tsv file containing [Workflow ID, sample_id, gvcf]",
    )
    parser.add_argument(
        "--relevant_coords",
        required=True,
        help="bed file with relevant analysis coordinates",
    )
    parser.add_argument(
        "--ground_truth_vcf",
        required=True,
        help="vcf file containing ground_truth genotypes for training-set",
    )
    parser.add_argument(
        "--processes", default=5, type=int, help="number of parallel processes to run"
    )

    parser.add_argument("-fc", help="index of first command", default=0)
    parser.add_argument("-lc", help="index of last command", default=1000)
    parser.add_argument("-d", action="store_true", help="print only")
    args = parser.parse_args(argv[1:])
    return args


def run(argv):
    """
    SEC (Systematic Error Correction) training pipeline
    """
    args = parse_args(argv)
    relevant_coords_file = args.relevant_coords
    out_dir = args.out_dir
    ground_truth_vcf = args.ground_truth_vcf
    processes = args.processes

    inputs_table = read_sec_pipelines_inputs_table(args.inputs_table)
    sample_ids = list(inputs_table["sample_id"])
    gvcf_files = list(inputs_table["gvcf"])

    sp = SimplePipeline(
        start=args.fc, end=args.lc, debug=args.d, output_stream=sys.stdout
    )

    training_commands = []
    training_file_per_sample = []

    os.makedirs(f"{out_dir}/allele_distributions", exist_ok=True)
    model_prefix = f"{out_dir}/conditional_allele_distribution"

    # Extract relevant_coords from remote GCP gvcf files, and save in local storage
    relevant_gvcf_files = \
        extract_relevant_gvcfs(
            sample_ids=sample_ids,
            gvcf_files=gvcf_files,
            out_dir=out_dir,
            relevant_coords_file=relevant_coords_file,
            sp=sp,
            processes=processes)

    # Generate error_correction_training commands per sample
    # These will count empirical allele distributions per training sample
    for sample_id, relevant_gvcf in zip(sample_ids, relevant_gvcf_files):
        allele_distributions = (
            f"{out_dir}/allele_distributions/{sample_id}{FileExtension.TSV.value}"
        )
        training_commands.append(
            f"python {ugvc_pkg}/pipelines/sec/error_correction_training.py "
            f"--relevant_coords {relevant_coords_file} "
            f"--ground_truth_vcf {ground_truth_vcf} "
            f"--gvcf {relevant_gvcf} "
            f"--sample_id {sample_id} "
            f"--output_file {allele_distributions}"
        )
        training_file_per_sample.append(allele_distributions)

    # Execute  error_correction_training commands
    sp.run_parallel(training_commands, max_num_of_processes=args.processes)

    # Write a config file pointing to each sample's allele distribution file
    training_files_file = f"{out_dir}/conditional_allele_distribution_files.txt"
    with open(training_files_file, "w") as fh:
        for training_file in training_file_per_sample:
            fh.write(f"{training_file}\n")

    # Aggregate empirical allele distributions of training-set
    sp.print_and_run(
        f"python {ugvc_pkg}/pipelines/sec/merge_conditional_allele_distributions.py "
        f"--conditional_allele_distribution_files {training_files_file} "
        f"--output_prefix {model_prefix}"
    )


if __name__ == "__main__":
    run(sys.argv)
