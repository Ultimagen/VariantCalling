#!/env/python
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

    inputs_table = read_sec_pipelines_inputs_table(args)
    sample_ids = list(inputs_table["sample_id"])
    gvcf_files = list(inputs_table["gvcf"])

    sp = SimplePipeline(
        start=args.fc, end=args.lc, debug=args.d, output_stream=sys.stdout
    )

    training_commands = []
    test_commands = []
    assess_commands = []
    training_file_per_sample = []

    os.makedirs(f"{out_dir}/allele_distributions", exist_ok=True)
    model_prefix = f"{out_dir}/conditional_allele_distribution"

    relevant_gvcf_files = \
        extract_relevant_gvcfs(
            sample_ids=sample_ids,
            gvcf_files=gvcf_files,
            out_dir=out_dir,
            relevant_coords_file=relevant_coords_file,
            sp=sp,
            processes=processes)

    for sample_id, relevant_gvcf in zip(sample_ids, relevant_gvcf_files):
        allele_distributions = (
            f"{out_dir}/allele_distributions/{sample_id}{FileExtension.TSV.value}"
        )
        training_commands.append(
            f"python {ugvc_pkg} error_correction_training "
            f"--relevant_coords {relevant_coords_file} "
            f"--ground_truth_vcf {ground_truth_vcf} "
            f"--gvcf {relevant_gvcf} "
            f"--sample_id {sample_id} "
            f"--output_file {allele_distributions}"
        )
        training_file_per_sample.append(allele_distributions)

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
