from __future__ import annotations

import pysam

# Utilities for working with VCF file records

DNA_SYMBOLS = ["A", "T", "G", "C"]


def is_symbolic(allele: str) -> bool:
    """Check if allele is symbolic

    Parameters
    ----------
    allele : str
        Input allele

    Returns
    -------
    bool
        T/F
    """
    return len([x for x in allele if x not in DNA_SYMBOLS]) > 0


def is_indel(rec: pysam.VariantRecord) -> list[bool]:
    """Checks for each alt allele of the variant if it is indel

    Parameters
    ----------
    rec : pysam.VariantRecord
        Input variant

    Returns
    -------
    List[bool]
        list of T/F
    """
    return [len(x) != len(rec.alleles[0]) and not is_symbolic(x) for x in rec.alleles]


def is_deletion(rec: pysam.VariantRecord) -> list[bool]:
    """For every allele in the variant record - report if the allele is deletion

    Parameters
    ----------
    rec : pysam.VariantRecord
        Report if the variant is deletion for each allele

    Returns
    -------
    List[bool]
        List of T/F
    """
    return [len(x) < len(rec.alleles[0]) and not is_symbolic(x) for x in rec.alleles]


def is_insertion(rec: pysam.VariantRecord) -> list[bool]:
    """For every allele in the variant record - report if the allele is insertion

    Parameters
    ----------
    rec : pysam.VariantRecord
        Report if the variant is deletion for each allele

    Returns
    -------
    List[bool]
        List of T/F
    """
    return [len(x) > len(rec.alleles[0]) and not is_symbolic(x) for x in rec.alleles]


def indel_length(rec: pysam.VariantRecord) -> list[int]:
    """Length of all indel alleles

    Parameters
    ----------
    rec : pysam.VariantRecord
        Input record

    Returns
    -------
    List[int]
        List of length differences between the allele and the reference allele
    """
    return [abs(len(x) - len(rec.alleles[0])) if not is_symbolic(x) else 0 for x in rec.alleles]
