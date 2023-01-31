from __future__ import annotations

from pysam.libcbcf import VariantRecord, VariantRecordSample

default_filter = ["<NON_REF>"]


def get_alleles_str(variant: VariantRecord) -> str:
    """
    Get a string representing the possible alleles of a variant

    Parameters
    ----------
    variant : VariantRecord
        a pysam VariantRecord

    Returns
    -------
    comma joined alleles string,
        e.g: ref=A alt=G.*
        get_alleles_str(v) = 'A,G,*'
    """
    return ",".join(variant.alleles)


def get_filtered_alleles_list(variant: VariantRecord, filter_list: list[str] = None) -> list[str]:
    """
    Get a filtered list of alleles of a variant

    Parameters
    ----------
    variant : VariantRecord
        a pysam VariantRecord
    filter_list : list[str]
        a list of alleles to be filtered, initialized to default_filter (global var)

    Returns
    -------
    list of alleles, excluding the filter_list (initialized to default_filter), and * as minor allele
        e.g: ref=A alt=G,* filter_list=['*']
             get_alleles_str(v) = ['A','G']
    """
    if filter_list is None:
        filter_list = default_filter
    filtered_alleles = [a for a in variant.alleles if a not in filter_list]

    # discard minor allele '*'
    if filtered_alleles[-1] == "*":
        filtered_alleles = filtered_alleles[:-1]
    return filtered_alleles


def get_filtered_alleles_str(variant: VariantRecord, filter_list: list[str] = None) -> str:
    """
    Get a filtered list of alleles of a variant (as string)

    Parameters
    ----------
    variant : VariantRecord
        a pysam VariantRecord
    filter_list : list[str]
        a list of alleles to be filtered, initialized to default_filter (global var)

    Returns
    -------
    Comma joined alleles string, excluding the filter_list (initialized to default_filter), and * as minor allele
        e.g: ref=A alt=G,* filter_list=['*']
             get_alleles_str(v) = 'A,G'
    """
    return ",".join(get_filtered_alleles_list(variant, filter_list))


def get_genotype(variant_record_sample: VariantRecordSample) -> str:
    """
    Get genotype string for a specific sample in a specific variant.
    e.g: ref=A, alt=T, GT=0/1 -> 'A/T'

    Parameters
    ----------
    variant_record_sample: VariantRecordSample
        a pysam sample record of a variant (genotype, AD, GQ, etc)

    Returns
    -------
    genotype of sample
    """
    alleles = ["." if a is None else str(a) for a in variant_record_sample.alleles]
    return "/".join(alleles)


def get_genotype_indices(variant_record_sample: VariantRecordSample) -> str:
    """
    Get genotype indices string for a specific sample in a specific variant.
    e.g: ref=A, alt=T, GT=0/1 -> '0/1'

    Parameters
    ----------
    variant_record_sample: VariantRecordSample
        a pysam sample record of a variant (genotype, AD, GQ, etc)

    Returns
    -------
    genotype indices of sample as string
    """
    allele_indices = ["." if a is None else str(a) for a in variant_record_sample.allele_indices]
    return "/".join(sorted(allele_indices))


def has_candidate_alternatives(variant: VariantRecord) -> bool:
    """
    Returns
    -------
    True iff position has a candidate alternative allele (besides <NON_REF>)
    """
    candidate_alleles_str = get_filtered_alleles_str(variant)
    return "," in candidate_alleles_str


def is_snp(alleles: list[str]):
    """
    Return True if alleles represent a SNP
    - If the SNP locus is also within a deletion (* allele), still return True.
    - Ignored <NON_REF> allele

    Parameters
    ----------
    alleles : list[str]
        list of alleles

    Returns
    -------
    True iff the alleles represent SNP locus (all are of size 1)
    """
    __alleles = [a for a in alleles if a != "<NON_REF>"]
    return len(__alleles) >= 2 and all(len(a) == 1 for a in __alleles)


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


def is_indel(rec: VariantRecord) -> list[bool]:
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


def is_deletion(rec: VariantRecord) -> list[bool]:
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


def is_insertion(rec: VariantRecord) -> list[bool]:
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


def indel_length(rec: VariantRecord) -> list[int]:
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
