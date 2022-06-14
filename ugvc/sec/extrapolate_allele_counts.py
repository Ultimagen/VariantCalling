from __future__ import annotations

from ugvc.sec.systematic_error_correction_record import SECRecord


def extrapolate_allele_counts(sec_records: list[SECRecord]):
    het_records = []
    hom_records = []
    for sec_record in sec_records:
        # don't extrapolate haploid loci
        if "/" not in sec_record.conditioned_genotype:
            return
        conditioned_genotype_alleles = sec_record.conditioned_genotype.split("/")
        is_hom = conditioned_genotype_alleles[0] == conditioned_genotype_alleles[1]
        if ">" not in sec_record.conditioned_genotype:
            pass
        elif is_hom:
            hom_records.append(sec_record)
        else:
            het_records.append(sec_record)
