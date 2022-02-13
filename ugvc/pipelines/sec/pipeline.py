import argparse
import os
import random
import sys

import pandas as pd

from simppl.simple_pipeline import SimplePipeline

from ugvc import base_dir


"""
SEC (Systematic Error Correction) training and validation pipeline
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--inputs_table', required=True,
                        help='A tsv file containing [Workflow ID, sample_id, gvcf, train_test_split')
    parser.add_argument('--relevant_coords', required=True, help='bed file with relevant analysis coordinates')
    parser.add_argument('--ground_truth_vcf', required=True,
                        help='vcf file containing ground_truth genotypes for training-set')
    parser.add_argument('--train_test_split', default=0.8, type=float,
                        help='how much samples to use for training')
    parser.add_argument('--processes', default=5, type=int, help='number of parallel processes to run')
    parser.add_argument('--novel_detection_only', default=False, action='store_true',
                        help='do not use information on known variants')
    parser.add_argument('-fc', help='index of first command', default=0)
    parser.add_argument('-lc', help='index of last command', default=1000)
    parser.add_argument('-d', action='store_true', help='print only')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    relevant_coords_file = args.relevant_coords
    out_dir = args.out_dir
    ground_truth_vcf = args.ground_truth_vcf

    sec_main = f'{base_dir}/pipelines/sec'

    novel_detection_only = args.novel_detection_only
    novel_detection_suffix = '_novel' if novel_detection_only else ''

    scripts_dir = f'{base_dir}/../src/bash'

    inputs_table = pd.read_csv(args.inputs_table, sep='\t')
    sample_counts = inputs_table['sample_id'].value_counts()
    duplicated_samples = sample_counts[sample_counts > 1]
    if duplicated_samples.shape[0] > 0:
        raise RuntimeError(f'Found duplicated sample_ds in input:\n{duplicated_samples}')
    sample_ids = list(inputs_table['sample_id'])
    gvcf_files = list(inputs_table['gvcf'])

    sp = SimplePipeline(start=args.fc, end=args.lc, debug=args.d, output_stream=sys.stdout)

    extract_variants_commands = []
    remove_duplicates_commands = []
    training_commands = []
    test_commands = []
    assess_commands = []
    training_file_per_sample = []

    authorize_gcp_command = 'export GCS_OAUTH_TOKEN=`gcloud auth application-default print-access-token`'
    os.makedirs(f'{out_dir}/gvcf', exist_ok=True)
    os.makedirs(f'{out_dir}/allele_distributions', exist_ok=True)
    os.makedirs(f'{out_dir}/correction{novel_detection_suffix}', exist_ok=True)
    os.makedirs(f'{out_dir}/assessment{novel_detection_suffix}', exist_ok=True)

    shuffled_samples_ids = sample_ids.copy()
    random.Random(42).shuffle(shuffled_samples_ids)
    training_samples = set(shuffled_samples_ids[:int(len(sample_ids) * args.train_test_split)])

    relevant_chromosomes = list(pd.read_csv(relevant_coords_file, sep='\t', header=None)[0].unique())

    model_file = f'{out_dir}/conditional_allele_distribution.pkl'

    for sample_id, gvcf_file in zip(sample_ids, gvcf_files):
        relevant_gvcf = f'{out_dir}/gvcf/{sample_id}.g.vcf'
        allele_distributions = f'{out_dir}/allele_distributions/{sample_id}.tsv'

        if not os.path.exists(relevant_gvcf):
            extract_variants_commands.append(f'{authorize_gcp_command}; '
                                             f'bcftools view {gvcf_file} {" ".join(relevant_chromosomes)} -Oz | '
                                             f'bedtools intersect -a stdin -b {relevant_coords_file} -header'
                                             f'> {relevant_gvcf}')

        remove_duplicates_commands.append(f'sh {scripts_dir}/remove_vcf_duplicates.sh {relevant_gvcf}')
        nodup_gvcf = f'{out_dir}/gvcf/{sample_id}.g.vcf.nodup.vcf.gz'

        if sample_id in training_samples:
            training_commands.append(f'python {sec_main}/error_correction_training.py '
                                     f'--relevant_coords {relevant_coords_file} '
                                     f'--ground_truth_vcf {ground_truth_vcf} '
                                     f'--gvcf {nodup_gvcf} '
                                     f'--sample_id {sample_id} '
                                     f'--output_file {allele_distributions}')
            training_file_per_sample.append(allele_distributions)
        else:
            corrected_vcf = f'{out_dir}/correction{novel_detection_suffix}/{sample_id}.vcf.gz'
            if novel_detection_only:
                test_commands.append(f'python {sec_main}/correct_systematic_errors.py '
                                     f'--relevant_coords {relevant_coords_file} '
                                     f'--model {model_file} '
                                     f'--gvcf {nodup_gvcf} '
                                     f'--output_file {corrected_vcf} '
                                     '--novel_detection_only')
            else:
                test_commands.append(f'python {sec_main}/correct_systematic_errors.py '
                                     f'--relevant_coords {relevant_coords_file} '
                                     f'--model {model_file} '
                                     f'--gvcf {nodup_gvcf} '
                                     f'--output_file {corrected_vcf}')

            assess_commands.append(f'python {base_dir}/concordance/compare_vcf_to_ground_truth.py '
                                   f'--relevant_coords {relevant_coords_file} '
                                   f'--ground_truth_vcf {ground_truth_vcf} '
                                   f'--gvcf {corrected_vcf} '
                                   f'--sample_id {sample_id} '
                                   f'--ignore_genotype '
                                   f'--output_prefix {out_dir}/assessment{novel_detection_suffix}/{sample_id}')

    # extract variants in relevant coords (often from cloud to local storage)
    sp.run_parallel(extract_variants_commands, max_num_of_processes=args.processes)

    # remove duplicate vcf records with disrupts indexing (is this because of overlap in relevant coords?)
    try:
        sp.run_parallel(remove_duplicates_commands, max_num_of_processes=args.processes)
    except RuntimeError:
        pass

    # Count empirical allele distributions per training sample
    sp.run_parallel(training_commands, max_num_of_processes=args.processes)

    training_files_file = f'{out_dir}/conditional_allele_distribution_files.txt'

    with open(training_files_file, 'w') as fh:
        for training_file in training_file_per_sample:
            fh.write(f'{training_file}\n')

    # Aggregate empirical allele distributions of training-set
    sp.print_and_run(
        f'python {sec_main}/merge_conditional_allele_distributions.py '
        f'--conditional_allele_distribution_files {training_files_file} '
        f'--output_file {model_file}')

    sp.run_parallel(test_commands, max_num_of_processes=args.processes)
    sp.run_parallel(assess_commands, max_num_of_processes=args.processes)


if __name__ == '__main__':
    main()
