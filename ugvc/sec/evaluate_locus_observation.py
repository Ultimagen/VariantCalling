from __future__ import annotations

from pysam import VariantRecord

from ugvc.sec.allele_counter import count_alleles_in_gvcf
from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.extrapolate_allele_counts import extrapolate_allele_counts
from ugvc.sec.systematic_error_correction_record import SECRecord
from ugvc.vcfbed.pysam_utils import get_filtered_alleles_str


def evaluate_observation(
    observed_variant: VariantRecord,
    expected_distribution: ConditionalAlleleDistribution,
) -> list[SECRecord]:
    """
    Evaluate likelihood and p-values of each conditioned genotype independently
    """
    sec_records = []
    chrom = observed_variant.chrom
    pos = observed_variant.pos
    observed_alleles = get_filtered_alleles_str(observed_variant)
    actual_allele_counts = count_alleles_in_gvcf(observed_variant)
    allele_counts_dict = expected_distribution.allele_counts_dict if expected_distribution is not None else {}

    for conditioned_genotype in allele_counts_dict:
        sec_records.append(
            SECRecord(
                chrom=chrom,
                pos=pos,
                expected_distribution=expected_distribution,
                conditioned_genotype=conditioned_genotype,
                observed_alleles=observed_alleles,
                actual_allele_counts=actual_allele_counts,
            )
        )

    # missing reference conditioned-genotype observation
    if "0/0" not in allele_counts_dict:
        conditioned_genotype = "0/0"
        expected_distribution = ConditionalAlleleDistribution(
            observed_alleles,
            conditioned_genotype,
            observed_alleles=observed_alleles,
            allele_counts={},
            num_of_samples_with_observed_alleles=0,
        )
        sec_records.append(
            SECRecord(
                chrom=chrom,
                pos=pos,
                expected_distribution=expected_distribution,
                conditioned_genotype="0/0",
                observed_alleles=observed_alleles,
                actual_allele_counts=actual_allele_counts,
            )
        )

    if len(sec_records) > 1:
        extrapolate_allele_counts(sec_records)

    for sec_record in sec_records:
        sec_record.process()

    return sec_records
