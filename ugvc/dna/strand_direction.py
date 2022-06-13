from enum import Enum


class StrandDirection(Enum):
    """
    A representation of strand direction
    """

    FORWARD = 1
    REVERSE = 0
    UNKNOWN = -1


def is_forward_strand(allele: str):
    """
    pileup alleles with uppercase letters represent forward strand
    todo - handle indels
    """
    return any(x in allele for x in ("A", "C", "G", "T"))
