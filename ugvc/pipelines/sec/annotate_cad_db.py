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
#    Manipulation on conditional error distributions
# CHANGELOG in reverse chronological order

import argparse
import itertools
import pickle

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.conditional_allele_distribution_correlator import correlate_distributions_per_pos
from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from ugvc.vcfbed.pysam_utils import is_snp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="path to pickle file containing conditional allele distributions per position of interest",
    )
    parser.add_argument(
        "--min_gt_correlation",
        default=0.99,
        type=float,
        help="don't call loci with lower correlation between ground-truth and observed genotypes",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.model, "rb") as file_handle:
        cad: ConditionalAlleleDistributions = pickle.load(file_handle)
        annotate_cad_db(cad, args.min_gt_correlation)


def annotate_cad_db(
    conditional_allele_distributions: ConditionalAlleleDistributions,
    min_gt_correlation: float,
):
    for (
        chrom,
        distributions_per_pos,
    ) in conditional_allele_distributions.distributions_per_chromosome.items():
        for pos, cad in distributions_per_pos.items():
            # inspect for a miss-represented ground-truth SNP
            if is_missing_ground_truth_snps(cad):
                print_annotation(chrom, pos, cad, "missing_snp")

            # inspect for uncorrelated ground-truth and observed genotypes
            max_correlation = correlate_distributions_per_pos(cad)
            if max_correlation < min_gt_correlation:
                print_annotation(chrom, pos, cad, "uncorrelated")

            elif is_hard_to_distinguish(cad):
                print_annotation(chrom, pos, cad, "hard_to_distinguish")


def print_annotation(
    chrom: str,
    pos: int,
    conditional_allele_distribution: ConditionalAlleleDistribution,
    annotation: str,
):
    conditioned_alleles = conditional_allele_distribution.conditioned_alleles
    for conditioned_genotype in conditional_allele_distribution.num_of_samples_with_alleles:
        for observed_alleles in conditional_allele_distribution.num_of_samples_with_alleles[conditioned_genotype]:
            allele_counts_str = conditional_allele_distribution.get_allele_counts_string(
                conditioned_genotype, observed_alleles
            )
            count = conditional_allele_distribution.num_of_samples_with_alleles[conditioned_genotype][observed_alleles]
            print(
                f"{chrom}\t{pos - 1}\t{pos}\t{annotation}\t{conditioned_alleles}\t{conditioned_genotype}\t"
                f"{observed_alleles}\t{count}\t{allele_counts_str}"
            )


def is_hard_to_distinguish(
    cad: ConditionalAlleleDistribution,
) -> bool:
    """
    Parameters
    ----------
    cad - ConditionalAlleleDistribution object

    Returns
    -------
    whether this cad represents a noisy position where the noise looks like a known common variant
    """
    gt_alleles = cad.conditioned_alleles.split(",")
    if len(gt_alleles) == 1:
        return False
    possible_genotype_counts = {}
    for possible_genotype in itertools.combinations(gt_alleles, 2):
        possible_genotype_counts[",".join(possible_genotype)] = 0

    for (
        conditioned_genotype,
        distributions_given_genotype,
    ) in cad.num_of_samples_with_alleles.items():
        if conditioned_genotype in {"1/1", "2/2"}:
            continue
        genotype_indices = conditioned_genotype.split("/")
        if "." in genotype_indices:
            continue
        for observed_alleles in distributions_given_genotype:
            if observed_alleles in possible_genotype_counts:
                possible_genotype_counts[observed_alleles] += 1

    if max(possible_genotype_counts.values()) > 1:
        return True
    return False


def is_missing_ground_truth_snps(cad: ConditionalAlleleDistribution) -> bool:
    for (
        conditioned_genotype,
        distribution_given_genotype,
    ) in cad.num_of_samples_with_alleles.items():
        if conditioned_genotype == "0/0":
            continue
        conditioned_alleles = cad.conditioned_alleles.split(",")
        genotype_indices = conditioned_genotype.split("/")
        if "." in genotype_indices:
            continue
        genotype_alleles = [conditioned_alleles[int(genotype_index)] for genotype_index in genotype_indices]
        if "*" in genotype_alleles:
            continue
        if is_snp(genotype_alleles):
            observed_gt_alleles = False
            for observed_alleles in distribution_given_genotype.num_of_samples_with_alleles:
                if all(a in observed_alleles for a in genotype_alleles):
                    observed_gt_alleles = True
            if not observed_gt_alleles:
                return True
    return False


if __name__ == "__main__":
    main()
