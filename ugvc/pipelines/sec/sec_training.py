#!/env/python
import os
import random
import sys

import pandas as pd
from simppl.cli import get_parser
from simppl.simple_pipeline import SimplePipeline

from ugvc import base_dir as ugvc_pkg
from ugvc.utils.consts import FileExtension


def parse_args(argv):
    parser = get_parser("sec_training", run.__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--inputs_table",
        required=True,
        help="A tsv file containing [Workflow ID, sample_id, gvcf, train_test_split",
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
        "--train_test_split",
        default=0.99,
        type=float,
        help="how much samples to use for training",
    )
    parser.add_argument(
        "--processes", default=5, type=int, help="number of parallel processes to run"
    )
    parser.add_argument(
        "--use_known_variants_info",
        default=False,
        action="store_true",
        help="use information on known variants",
    )
    parser.add_argument("-fc", help="index of first command", default=0)
    parser.add_argument("-lc", help="index of last command", default=1000)
    parser.add_argument("-d", action="store_true", help="print only")
    args = parser.parse_args(argv[1:])
    return args


def run(argv):
    """
    SEC (Systematic Error Correction) training and validation pipeline
    """
    args = parse_args(argv)
    relevant_coords_file = args.relevant_coords
    out_dir = args.out_dir
    ground_truth_vcf = args.ground_truth_vcf

    novel_detection_only = not args.use_known_variants_info
    novel_detection_suffix = "_novel" if novel_detection_only else ""

    inputs_table = pd.read_csv(args.inputs_table, sep="\t")
    sample_counts = inputs_table["sample_id"].value_counts()
    duplicated_samples = sample_counts[sample_counts > 1]
    if duplicated_samples.shape[0] > 0:
        raise RuntimeError(
            f"Found duplicated sample_ds in input:\n{duplicated_samples}"
        )
    sample_ids = list(inputs_table["sample_id"])
    gvcf_files = list(inputs_table["gvcf"])

    sp = SimplePipeline(
        start=args.fc, end=args.lc, debug=args.d, output_stream=sys.stdout
    )

    extract_variants_commands = []
    index_chr_vcfs_commands = []
    concat_vcf_commands = []
    index_vcf_commands = []
    training_commands = []
    test_commands = []
    assess_commands = []
    training_file_per_sample = []

    authorize_gcp_command = (
        "export GCS_OAUTH_TOKEN=`gcloud auth application-default print-access-token`"
    )
    os.makedirs(f"{out_dir}/gvcf", exist_ok=True)
    os.makedirs(f"{out_dir}/allele_distributions", exist_ok=True)
    os.makedirs(f"{out_dir}/correction{novel_detection_suffix}", exist_ok=True)
    os.makedirs(f"{out_dir}/assessment{novel_detection_suffix}", exist_ok=True)

    shuffled_samples_ids = sample_ids.copy()
    random.Random(42).shuffle(shuffled_samples_ids)
    training_samples = set(
        shuffled_samples_ids[: int(len(sample_ids) * args.train_test_split)]
    )

    relevant_chromosomes = list(
        pd.read_csv(relevant_coords_file, sep="\t", header=None)[0].unique()
    )

    model_prefix = f"{out_dir}/conditional_allele_distribution"

    for sample_id, gvcf_file in zip(sample_ids, gvcf_files):
        relevant_gvcf = f"{out_dir}/gvcf/{sample_id}.g.vcf.gz"
        allele_distributions = (
            f"{out_dir}/allele_distributions/{sample_id}{FileExtension.TSV.value}"
        )

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

        if sample_id in training_samples:
            training_commands.append(
                f"python {ugvc_pkg} error_correction_training "
                f"--relevant_coords {relevant_coords_file} "
                f"--ground_truth_vcf {ground_truth_vcf} "
                f"--gvcf {relevant_gvcf} "
                f"--sample_id {sample_id} "
                f"--output_file {allele_distributions}"
            )
            training_file_per_sample.append(allele_distributions)
        else:
            corrected_vcf = (
                f"{out_dir}/correction{novel_detection_suffix}/{sample_id}.vcf.gz"
            )
            if novel_detection_only:
                test_commands.append(
                    f"python {ugvc_pkg} correct_systematic_errors "
                    f"--relevant_coords {relevant_coords_file} "
                    f"--model {model_prefix}*.pkl "
                    f"--gvcf {relevant_gvcf} "
                    f"--output_file {corrected_vcf} "
                    "--novel_detection_only"
                )
            else:
                test_commands.append(
                    f"python {ugvc_pkg} correct_systematic_errors "
                    f"--relevant_coords {relevant_coords_file} "
                    f"--model \"{model_prefix}*.pkl\" "
                    f"--gvcf {relevant_gvcf} "
                    f"--output_file {corrected_vcf}"
                )

    # extract variants in relevant coords (often from cloud to local storage)
    sp.run_parallel(extract_variants_commands, max_num_of_processes=args.processes)
    sp.run_parallel(index_chr_vcfs_commands, max_num_of_processes=args.processes)

    # concat vcf per chromosome and remove duplicate vcf records with disrupts indexing
    # (is this because of overlap in relevant coords?)
    try:
        sp.run_parallel(concat_vcf_commands, max_num_of_processes=args.processes)
        sp.run_parallel(index_vcf_commands, max_num_of_processes=args.processes)
    except RuntimeError:
        pass

    # Count empirical allele distributions per training sample
    sp.run_parallel(training_commands, max_num_of_processes=args.processes)

    training_files_file = f"{out_dir}/conditional_allele_distribution_files.txt"

    with open(training_files_file, "w") as fh:
        for training_file in training_file_per_sample:
            fh.write(f"{training_file}\n")

    # Aggregate empirical allele distributions of training-set
    sp.print_and_run(
        f"python {ugvc_pkg} merge_conditional_allele_distributions "
        f"--conditional_allele_distribution_files {training_files_file} "
        f"--output_prefix {model_prefix}"
    )

    sp.run_parallel(test_commands, max_num_of_processes=args.processes)
    sp.run_parallel(assess_commands, max_num_of_processes=args.processes)


if __name__ == "__main__":
    run(sys.argv)
