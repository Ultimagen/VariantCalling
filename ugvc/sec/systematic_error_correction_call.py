from enum import Enum
from typing import List, Optional

from ugvc.sec.systematic_error_correction_record import SECRecord


class SECCallType(Enum):
    novel = "novel"
    known = "known"
    reference = "reference"
    uncorrelated = "uncorrelated"
    non_noise_allele = "non_noise_allele"
    unobserved = "unobserved"


class SECCall:
    def __init__(
        self,
        call_type: SECCallType,
        alleles: str,
        genotype: str,
        sec_records: List[SECRecord],
        genotype_quality: Optional[int] = None,
        novel_variant_p_value: Optional[float] = None,
        gt_correlation: Optional[float] = None,
    ):
        self.call_type = call_type
        self.alleles = alleles
        self.genotype = genotype
        self.sec_records = sec_records
        self.genotype_quality = genotype_quality
        self.novel_variant_p_value = novel_variant_p_value
        self.gt_correlation = gt_correlation

    def get_alleles_list(self) -> List[str]:
        return self.alleles.split(",")

    def get_genotype_indices_tuple(self):
        return tuple([int(i) if i != "." else None for i in self.genotype.split("/")])

    def __str__(self):
        if self.call_type == SECCallType.reference:
            return f"call ref {self.genotype} gt_corr={self.gt_correlation} GQ={self.genotype_quality}"
        elif self.call_type == SECCallType.known:
            return f"call known variant {self.genotype} gt_corr={self.gt_correlation} GQ={self.genotype_quality}"
        elif self.call_type == SECCallType.uncorrelated:
            return (
                f"don't call uncorrelated variant {self.genotype} gt_corr={self.gt_correlation} "
                f"GQ={self.genotype_quality}"
            )
        else:
            return f"call {self.call_type} variant {self.alleles} {self.genotype}"
