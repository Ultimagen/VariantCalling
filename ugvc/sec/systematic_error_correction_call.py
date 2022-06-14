from __future__ import annotations

from enum import Enum

from ugvc.sec.systematic_error_correction_record import SECRecord


class SECCallType(Enum):
    NOVEL = "novel"
    KNOWN = "known"
    REFERENCE = "reference"
    UNCORRELATED = "uncorrelated"
    NON_NOISE_ALLELE = "non_noise_allele"
    UNOBSERVED = "unobserved"


class SECCall:
    def __init__(
        self,
        call_type: SECCallType,
        alleles: str,
        genotype: str,
        sec_records: list[SECRecord],
        genotype_quality: int | None = None,
        novel_variant_p_value: float | None = None,
        gt_correlation: float | None = None,
    ):
        self.call_type = call_type
        self.alleles = alleles
        self.genotype = genotype
        self.sec_records = sec_records
        self.genotype_quality = genotype_quality
        self.novel_variant_p_value = novel_variant_p_value
        self.gt_correlation = gt_correlation

    def get_alleles_list(self) -> list[str]:
        return self.alleles.split(",")

    def get_genotype_indices_tuple(self):
        # pylint: disable=consider-using-generator
        return tuple([int(i) if i != "." else None for i in self.genotype.split("/")])

    def __str__(self):
        if self.call_type == SECCallType.REFERENCE:
            return f"call ref {self.genotype} gt_corr={self.gt_correlation} GQ={self.genotype_quality}"
        if self.call_type == SECCallType.KNOWN:
            return f"call known variant {self.genotype} gt_corr={self.gt_correlation} GQ={self.genotype_quality}"
        if self.call_type == SECCallType.UNCORRELATED:
            return (
                f"don't call uncorrelated variant {self.genotype} gt_corr={self.gt_correlation} "
                f"GQ={self.genotype_quality}"
            )
        return f"call {self.call_type} variant {self.alleles} {self.genotype}"
