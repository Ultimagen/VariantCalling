from typing import List

from pysam.libcbcf import VariantRecord, VariantRecordSample

default_filter = ['<NON_REF>']


def get_alleles_str(variant: VariantRecord) -> str:
    """
    @param variant: a pysam variant object
    @return: comma joined alleles string,
        e.g: ref=A alt=G.*
        get_alleles_str(v) = 'A,G,*'
    """
    return ','.join(variant.alleles)


def get_filtered_alleles_list(variant: VariantRecord, filter_list: List[str] = None) -> List[str]:
    """
    @param variant: a variant
    @param filter_list: a list of alleles to filter
    @return: list of alleles,
        excluding the filter_list (initialized to default_filter), and * as minor allele
        e.g: ref=A alt=G,* filter_list=['*']
            get_alleles_str(v) = ['A','G']
    """
    if filter_list is None:
        filter_list = default_filter
    filtered_alleles = [a for a in variant.alleles if a not in filter_list]

    # discard minor allele '*'
    if filtered_alleles[-1] == '*':
        filtered_alleles = filtered_alleles[:-1]
    return filtered_alleles


def get_filtered_alleles_str(variant: VariantRecord, filter_list: List[str] = None) -> str:
    """
    @param variant: a variant
    @param filter_list: a list of alleles to filter
    @return: comma joined alleles string,
        excluding the filter_list (initialized to default_filter), and * as minor allele
         e.g: ref=A alt=G,* filter_list=['*']
              get_alleles_str(v) = ['A','G']
    """
    return ','.join(get_filtered_alleles_list(variant, filter_list))


def get_genotype(variant_record_sample: VariantRecordSample) -> str:
    """
    @param variant_record_sample: a sample record of a variant (genotype, AD, GQ, etc)
    @return: genotype of sample
        e.g: ref=A, alt=T, GT=0/1 -> 'A/T'
    """
    alleles = ['.' if a is None else str(a) for a in variant_record_sample.alleles]
    return '/'.join(alleles)


def get_genotype_indices(variant_record_sample: VariantRecordSample) -> str:
    """
     @param variant_record_sample: a sample record of a variant (genotype, AD, GQ, etc)
     @return: genotype of sample
         e.g: ref=A, alt=T, GT=0/1 -> '0/1'
     """
    allele_indices = ['.' if a is None else str(a) for a in variant_record_sample.allele_indices]
    return '/'.join(sorted(allele_indices))

def has_candidate_alternatives(variant: VariantRecord) -> bool:
    """
    Return True iff position has a candidate alternative allele (besides <NON_REF>)
    """
    candidate_alleles_str = get_filtered_alleles_str(variant)
    return ',' in candidate_alleles_str


def is_snp(alleles: List[str]):
    """
    @param alleles: list of alleles
    @return: are the alleles represent SNP locus (all are of size 1)
    """
    return all([len(a) == 1 for a in alleles])


def fix_ultima_info(variant: VariantRecord, vcf_header) -> None:
    """
    Some String typed fields are read by pysam as tuple, construct back the intended string
    @param variant: a pysam variant object
    @param vcf_header: header object of VariantFile
    """
    info = variant.info
    for key, value in info.items():
        if vcf_header.info.get(key).type == 'String':
            variant.info[key] = ','.join(value)
