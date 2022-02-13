from typing import List

from pysam.libcbcf import VariantRecord, VariantRecordSample

default_filter = ['<NON_REF>']


def get_alleles_str(variant: VariantRecord) -> str:
    return ','.join(variant.alleles)


def get_filtered_alleles_list(variant: VariantRecord, filter_list: List[str] = None) -> List[str]:
    if filter_list is None:
        filter_list = default_filter
    filtered_alleles = [a for a in variant.alleles if a not in filter_list]

    # discard minor allele '*'
    if filtered_alleles[-1] == '*':
        filtered_alleles = filtered_alleles[:-1]
    return filtered_alleles


def get_filtered_alleles_str(variant: VariantRecord, filter_list: List[str] = None) -> str:
    return ','.join(get_filtered_alleles_list(variant, filter_list))


def get_genotype(variant_record_sample: VariantRecordSample) -> str:
    alleles = ['.' if a is None else str(a) for a in variant_record_sample.alleles]
    return '/'.join(alleles)


def get_genotype_indices(variant_record_sample: VariantRecordSample) -> str:
    allele_indices = ['.' if a is None else str(a) for a in variant_record_sample.allele_indices]
    return '/'.join(sorted(allele_indices))


def get_extended_genotype(variant: VariantRecord,
                          variant_record_sample: VariantRecordSample,
                          keep_phasing_order: bool = False) -> str:
    """
    @param variant - the variant record (vcf line)
    @param  variant_record_sample - the info for the specific sample we're interested in
    @param keep_phasing_order - whether to report alleles in phased order or by order of alleles (ref first, etc.)
    Return genotype tuple, with encoded ref and alt alleles
    A ref allele is simply the ref base/s.
    An alt allele is encoded as ref>alt
    For example:
    ref=A, alt=AC genotype=0/1 -> will return (A, A>AC)
    """
    genotype = []
    genotype_indices = variant_record_sample['GT']
    if not keep_phasing_order and None not in genotype_indices:
        genotype_indices = sorted(genotype_indices)
    for genotype_index in genotype_indices:
        if genotype_index is None:
            allele = '.'
        else:
            allele = variant.alleles[genotype_index]
        genotype.append(get_extended_allele(allele, variant.ref))

    if keep_phasing_order:
        return '|'.join(genotype)
    else:
        return '/'.join(genotype)


def get_extended_allele(allele: str, ref: str):
    if allele == ref:
        return allele
    else:
        return f'{ref}>{allele}'


def is_ref_call(variant_record_sample: VariantRecordSample) -> bool:
    """
    Return True iff sample has reference genotype 0, 0/0, etc.
    """
    num_of_alt_alleles = 0
    for i in variant_record_sample.allele_indices:
        if i is not None and i > 0:
            num_of_alt_alleles += 1
    return num_of_alt_alleles == 0


def has_candidate_alternatives(variant: VariantRecord) -> bool:
    """
    Return True iff position has a candidate alternative allele (besides <NON_REF>)
    """
    candidate_alleles_str = get_filtered_alleles_str(variant)
    return ',' in candidate_alleles_str


def is_snp(alleles: List[str]):
    allele_lengths = {len(a) for a in alleles}
    return len(allele_lengths) == 1 and len(alleles[0]) == 1


def fix_ultima_info(variant: VariantRecord, vcf_header):
    """
    Some String typed fields are read by pysam as tuple, construct back the intended string
    """
    info = dict(variant.info)
    for key, value in info.items():
        if vcf_header.info.get(key).type == 'String':
            info[key] = ','.join(value)

