from typing import Dict

import numpy as np
from scipy.stats import chisquare, binom_test

from ugvc.dna.strand_direction import StrandDirection
from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution, get_allele_counts_list
from ugvc.sec.read_counts import ReadCounts
from ugvc.utils.stats_utils import scale_contingency_table, multinomial_likelihood_ratio
from ugvc.utils.pysam_utils import is_snp


class SECRecord:
    noise_ratio_for_unobserved_snps = 0.02
    noise_ratio_for_unobserved_indels = 0.05

    def __init__(self,
                 chrom: str,
                 pos: int,
                 expected_distribution: ConditionalAlleleDistribution,
                 conditioned_genotype: str,
                 observed_alleles: str,
                 actual_allele_counts: Dict[str, ReadCounts]):

        self.chrom = chrom
        self.pos = pos
        self.conditioned_alleles = expected_distribution.conditioned_alleles
        self.conditioned_genotype = conditioned_genotype
        self.observed_alleles = observed_alleles
        self.observed_alleles_list = observed_alleles.split(',')
        self.actual_allele_counts = actual_allele_counts
        self.expected_distribution = expected_distribution
        self.actual_distribution_list = get_allele_counts_list(actual_allele_counts, observed_alleles)
        self.were_observed_alleles_in_db = \
            observed_alleles in expected_distribution.get_possible_observed_alleles(conditioned_genotype) \
            and expected_distribution.num_of_samples_with_alleles[conditioned_genotype][observed_alleles] > 0
        if self.were_observed_alleles_in_db:
            self.expected_distribution_list = \
                expected_distribution.get_observed_alleles_counts_list(conditioned_genotype, observed_alleles)
        else:
            expected_distribution_arr = np.ones(2 * len(self.observed_alleles_list))
            if is_snp(self.observed_alleles_list):
                for i in (0, 1):
                    expected_distribution_arr[i] = 1 / self.noise_ratio_for_unobserved_snps
            else:
                for i in (0, 1):
                    expected_distribution_arr[i] = 1 / self.noise_ratio_for_unobserved_indels
            self.expected_distribution_list = list(expected_distribution_arr)

        self.num_of_observations_expected = sum(self.expected_distribution_list)
        self.num_of_observations_actual = sum(self.actual_distribution_list)

        self.scaled_expected_distribution_list = self.__scale_expected_distribution_list()

        self.observed_alleles_frequency = expected_distribution.get_observed_alleles_frequency(conditioned_genotype,
                                                                                               observed_alleles)
        self.likelihood = 0
        self.likelihood_ratio = 0
        self.chi2_pval = None
        self.forward_enrichment_pval = 1
        self.reverse_enrichment_pval = 1
        self.strand_enrichment_pval = 1
        self.__alt_enrichment_pval = 1
        self.frequency_scaled_strand_enrichment_pval = 1
        self.lesser_strand_enrichment_pval = 1

    def process(self):
        self.__calc_strand_enrichment()
        self.__calc_alt_enrichment()

        likelihood, likelihood_ratio = multinomial_likelihood_ratio(self.actual_distribution_list,
                                                                    self.expected_distribution_list)

        self.likelihood = likelihood * self.observed_alleles_frequency
        self.likelihood_ratio = likelihood_ratio * self.observed_alleles_frequency

        if min(self.scaled_expected_distribution_list) > 4 and min(self.actual_distribution_list) > 4:
            chi_sq, self.chi2_pval = chisquare(f_obs=self.actual_distribution_list,
                                               f_exp=self.scaled_expected_distribution_list)

    def __calc_strand_enrichment(self):
        self.forward_enrichment_pval = self.__single_strand_binomial_test(StrandDirection.FORWARD)
        self.reverse_enrichment_pval = self.__single_strand_binomial_test(StrandDirection.REVERSE)
        # each strand is tested independently
        self.strand_enrichment_pval = self.forward_enrichment_pval * self.reverse_enrichment_pval

        # scale p-value by observed_alleles_frequency
        self.frequency_scaled_strand_enrichment_pval = \
            self.observed_alleles_frequency * self.strand_enrichment_pval

        self.lesser_strand_enrichment_pval = max(self.forward_enrichment_pval,
                                                 self.reverse_enrichment_pval)

    def __calc_alt_enrichment(self):
        expected_alt = sum(self.expected_distribution_list[2:])
        actual_alt = sum(self.actual_distribution_list[2:])
        if self.num_of_observations_expected > 0:
            p_alt = (expected_alt + 1) / (self.num_of_observations_expected + 2)
        else:
            if is_snp(self.observed_alleles.split(',')):
                p_alt = self.noise_ratio_for_unobserved_snps
            else:
                p_alt = self.noise_ratio_for_unobserved_indels

        self.__alt_enrichment_pval = \
            binom_test(actual_alt, n=self.num_of_observations_actual, p=p_alt, alternative='greater')

    def __scale_expected_distribution_list(self):
        return scale_contingency_table(self.expected_distribution_list, self.num_of_observations_actual)

    def __single_strand_binomial_test(self, strand: StrandDirection) -> float:
        if strand == StrandDirection.FORWARD:
            ref_index = 0
            alt_index = 2
        else:
            ref_index = 1
            alt_index = 3

        expected_ref = self.expected_distribution_list[ref_index]
        actual_ref = self.actual_distribution_list[ref_index]

        # sum up all alternatives (since we want to reject null hypothesis := reference)
        expected_alt = sum(self.expected_distribution_list[alt_index::2])
        actual_alt = sum(self.actual_distribution_list[alt_index::2])

        expected_total = expected_ref + expected_alt
        actual_total = actual_alt + actual_ref

        if expected_total > 0:
            p_alt_f = (expected_alt + 1) / (expected_total + 2)  # add-one correction
        else:
            # when no reads on this strand were expected, look for moderate alt enrichment
            if is_snp(self.observed_alleles.split(',')):
                p_alt_f = self.noise_ratio_for_unobserved_snps
            else:
                p_alt_f = self.noise_ratio_for_unobserved_indels

        return binom_test(actual_alt, n=actual_total, p=p_alt_f, alternative='greater')

    def __str__(self):
        fields = [self.chrom,
                  str(self.pos),
                  self.conditioned_alleles,
                  self.conditioned_genotype,
                  self.observed_alleles,
                  f'actual={self.actual_distribution_list}',
                  f'expected={self.scaled_expected_distribution_list}',
                  f'OAF={"%.2g" % self.observed_alleles_frequency}',
                  f'L={"%.2g" % self.likelihood}',
                  f'f_pval={"%.2g" % self.forward_enrichment_pval}',
                  f'r_pval={"%.2g" % self.reverse_enrichment_pval}',
                  f'strand_pval={"%.2g" % self.strand_enrichment_pval}',
                  f'alt_pval={"%.2g" % self.__alt_enrichment_pval}',
                  f'LR={"%.2g" % self.likelihood_ratio}',
                  f'actual_allele_counts={self.actual_allele_counts}',
                  ]
        if self.chi2_pval is not None:
            fields.append(f'chi2.pval={"%.2g" % self.chi2_pval}')
        return '\t'.join(fields)

    def __repr__(self):
        return str(self)
