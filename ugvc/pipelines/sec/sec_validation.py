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
#    SEC validation pipeline
# CHANGELOG in reverse chronological order

import os
import sys

from simppl.cli import get_parser
from simppl.simple_pipeline import SimplePipeline

from ugvc import base_dir as ugvc_pkg
from ugvc.sec.sec_pipeline_utils import extract_relevant_gvcfs, read_sec_pipelines_inputs_table


def parse_args(argv):
    parser = get_parser("sec_validation", run.__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--inputs_table",
        required=True,
        help="A tsv file containing [Workflow ID, sample_id, gvcf, comp_h5]",
    )
    parser.add_argument(
        "--relevant_coords",
        required=True,
        help="bed file with relevant analysis coordinates",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="A glob pattern of conditional allele distributions pkl files",
    )
    parser.add_argument(
        "--genome_fasta",
        required=True,
        help="Reference genome file",
    )
    parser.add_argument("--processes", default=5, type=int, help="number of parallel processes to run")
    parser.add_argument(
        "--hcr",
        required=True,
        help="hcr for assess_sec_concordance (runs.convervative.bed)",
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
    SEC (Systematic Error Correction) validation pipeline
    """
    args = parse_args(argv)
    relevant_coords_file = args.relevant_coords
    out_dir = args.out_dir
    model = args.model
    processes = args.processes
    genome_fasta = args.genome_fasta
    novel_detection_only = not args.use_known_variants_info
    novel_detection_suffix = "_novel" if novel_detection_only else ""

    inputs_table = read_sec_pipelines_inputs_table(args.inputs_table)
    sample_ids = list(inputs_table["sample_id"])
    gvcf_files = list(inputs_table["gvcf"])
    comp_h5_files = list(inputs_table["comp_h5"])

    simple_pipline = SimplePipeline(start=args.fc, end=args.lc, debug=args.d, output_stream=sys.stdout)

    test_commands = []
    assess_commands = []

    os.makedirs(f"{out_dir}/correction{novel_detection_suffix}", exist_ok=True)
    os.makedirs(f"{out_dir}/assessment{novel_detection_suffix}", exist_ok=True)

    relevant_gvcf_files = extract_relevant_gvcfs(
        sample_ids=sample_ids,
        gvcf_files=gvcf_files,
        out_dir=out_dir,
        relevant_coords_file=relevant_coords_file,
        simple_pipeline=simple_pipline,
        processes=processes,
    )

    for sample_id, relevant_gvcf, comparison_table in zip(sample_ids, relevant_gvcf_files, comp_h5_files):
        sec_vcf = f"{out_dir}/correction{novel_detection_suffix}/{sample_id}.vcf.gz"

        if novel_detection_only:
            test_commands.append(
                f"python {ugvc_pkg} correct_systematic_errors "
                f"--relevant_coords {relevant_coords_file} "
                f'--model "{model}" '
                f"--gvcf {relevant_gvcf} "
                f"--output_file {sec_vcf} "
                "--novel_detection_only"
            )
        else:
            test_commands.append(
                f"python {ugvc_pkg} correct_systematic_errors "
                f"--relevant_coords {relevant_coords_file} "
                f'--model "{model}" '
                f"--gvcf {relevant_gvcf} "
                f"--output_file {sec_vcf}"
            )

        assess_commands.append(
            f"python {ugvc_pkg} assess_sec_concordance "
            f"--concordance_h5_input {comparison_table} "
            f"--genome_fasta {genome_fasta} "
            f"--raw_exclude_list {relevant_coords_file} "
            f"--sec_exclude_list {sec_vcf}.bed "
            f"--hcr {args.hcr} "
            f"--output_prefix {out_dir}/assessment{novel_detection_suffix}/{sample_id}"
        )

    simple_pipline.run_parallel(test_commands, max_num_of_processes=args.processes)
    simple_pipline.run_parallel(assess_commands, max_num_of_processes=args.processes)


if __name__ == "__main__":
    run(sys.argv)
