#!/env/python
import argparse
import os.path
import pickle
import sys
from typing import List

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from ugvc.sec.read_counts import ReadCounts


def get_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditional_allele_distribution_files",
        help="file containing paths to synced conditional allele distribution files (tsv)",
        required=True,
    )
    parser.add_argument(
        "--output_prefix",
        help="prefix to pickle files (per chromosome) "
        "of serialized ConditionalAlleleDistribution dicts",
        required=True,
    )
    args = parser.parse_args(argv[1:])
    return args


def run(argv: List[str]):
    """
    stack together conditional_allele_distributions from multiple samples into a single file
    """
    args = get_args(argv)

    conditional_allele_distributions = ConditionalAlleleDistributions()
    for file_name in open(args.conditional_allele_distribution_files):
        file_name = file_name.strip()
        if not os.path.exists(file_name):
            continue

        for line in open(file_name):
            (
                chrom,
                pos,
                ground_truth_alleles,
                true_genotype,
                observed_alleles,
                n_samples,
                allele_counts,
            ) = line.split("\t")
            pos = int(pos)
            allele_counts = allele_counts.split()
            alleles = allele_counts[0::2]
            counts = [
                ReadCounts(*[int(sc) for sc in c.split(",")])
                for c in allele_counts[1::2]
            ]
            allele_counts_dict = dict(zip(alleles, counts))
            conditional_allele_distributions.add_counts(
                chrom,
                pos,
                ConditionalAlleleDistribution(
                    ground_truth_alleles,
                    true_genotype,
                    observed_alleles,
                    allele_counts_dict,
                    int(n_samples),
                ),
            )

    with open(f"{args.output_prefix}.txt", "w") as otf:
        for (
            chrom,
            distributions_per_chrom,
        ) in conditional_allele_distributions.distributions_per_chromosome.items():
            for pos, conditional_allele_distribution in distributions_per_chrom.items():
                for record in conditional_allele_distribution.get_string_records(
                    chrom, pos
                ):
                    otf.write(f"{record}\n")
            with open(f"{args.output_prefix}.{chrom}.pkl", "wb") as out_pickle_file:
                pickle.dump(distributions_per_chrom, out_pickle_file)


if __name__ == "__main__":
    run(sys.argv)
