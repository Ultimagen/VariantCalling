from typing import List

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.systematic_error_correction_record import SECRecord
from ugvc.stats.goodness_of_fit import scale_contingency_table, multinomial_likelihood_ratio


def correlate_sec_records(sec_records: List[SECRecord], min_scale: int = 50) -> float:
    """
    Give high score to loci where two (matching) ground-truth genotypes yield about the same observed data.
    This suggests false ground-truth due to mapping issues.
    @precondition: all sec_records are for the same observed-alleles
    """
    expected_distribution_lists = []
    max_likelihood = max([sec_record.likelihood for sec_record in sec_records])
    for sec_record in sec_records:
        if sec_record.were_observed_alleles_in_db and sec_record.likelihood > max_likelihood / 1000:
            scale_to = min(sec_record.num_of_observations_expected, min_scale)
            scaled_distribution = scale_contingency_table(sec_record.expected_distribution_list, scale_to)
            expected_distribution_lists.append(scaled_distribution)

    return find_min_correlation(expected_distribution_lists)


def correlate_distributions_per_pos(cad: ConditionalAlleleDistribution) -> float:
    """
    Find the max-correlation per observed_alleles in a locus
    Return the max overall amongst all possible observed alleles
    """
    all_possible_observed_alleles = set()
    for conditioned_genotype in cad.num_of_samples_with_alleles:
        all_possible_observed_alleles.update(cad.get_possible_observed_alleles(conditioned_genotype))

    min_correlation = 1
    for observed_alleles in all_possible_observed_alleles:
        expected_distribution_lists = []
        for conditioned_genotype in cad.num_of_samples_with_alleles:
            if observed_alleles in cad.num_of_samples_with_alleles[conditioned_genotype]:
                expected_distribution_list = cad.get_observed_alleles_counts_list(conditioned_genotype,
                                                                                  observed_alleles)
                expected_distribution_lists.append(expected_distribution_list)
        min_correlation = min(find_min_correlation(expected_distribution_lists), min_correlation)
    return min_correlation


def find_min_correlation(expected_distribution_lists: List[List[int]]) -> float:
    max_likelihood_ratio = 0
    for i, expected_distribution_list_i in enumerate(expected_distribution_lists):
        if sum(expected_distribution_list_i) == 0:
            continue
        for j in range(i + 1, len(expected_distribution_lists)):
            expected_distribution_list_j = expected_distribution_lists[j]
            if sum(expected_distribution_list_j) == 0:
                continue
            likelihood, likelihood_ratio = multinomial_likelihood_ratio(expected_distribution_list_i,
                                                                        expected_distribution_list_j)

            max_likelihood_ratio = max(max_likelihood_ratio, likelihood_ratio)
    min_correlation = 1 - max_likelihood_ratio
    return min_correlation
