import argparse
import os.path
import pickle

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from ugvc.sec.read_counts import ReadCounts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conditional_allele_distribution_files',
        help='file containing paths to synced conditional allele distribution files (tsv)', required=True)
    parser.add_argument('--output_file', help='pickle file of serialized ConditionalAlleleDistributions', required=True)
    args = parser.parse_args()
    return args


def merge_conditional_allele_distributions():
    """
    stack together conditional_allele_distributions from multiple samples into a single file
    """
    args = get_args()

    conditional_allele_distributions = ConditionalAlleleDistributions()
    for file_name in open(args.conditional_allele_distribution_files):
        file_name = file_name.strip()
        if not os.path.exists(file_name):
            continue
        for line in open(file_name):
            chrom, pos, ground_truth_alleles, true_genotype, observed_alleles, n_samples, allele_counts = \
                line.split('\t')
            pos = int(pos)
            allele_counts = allele_counts.split()
            alleles = allele_counts[0::2]
            counts = [ReadCounts(*[int(sc) for sc in c.split(',')]) for c in allele_counts[1::2]]
            allele_counts_dict = dict(zip(alleles, counts))
            conditional_allele_distributions.add_counts(chrom, pos,
                                                        ConditionalAlleleDistribution(ground_truth_alleles,
                                                                                      true_genotype,
                                                                                      observed_alleles,
                                                                                      allele_counts_dict,
                                                                                      int(n_samples)))

    with open(f'{args.output_file}.txt', 'w') as otf:
        for chrom, distributions_per_chrom in conditional_allele_distributions.distributions_per_chromosome.items():
            for pos, conditional_allele_distribution in distributions_per_chrom.items():
                for record in conditional_allele_distribution.get_string_records(chrom, pos):
                    otf.write(f'{record}\n')
    with open(args.output_file, 'wb') as out_pickle_file:
        pickle.dump(conditional_allele_distributions, out_pickle_file)


if __name__ == '__main__':
    merge_conditional_allele_distributions()
