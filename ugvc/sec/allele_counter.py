from typing import Dict

from pysam import PileupColumn, VariantRecord

from ugvc.dna.strand_direction import is_forward_strand, StrandDirection
from ugvc.sec.read_counts import ReadCounts
from ugvc.utils.pysam_utils import get_filtered_alleles_list


def count_alleles_in_pileup(pc: PileupColumn) -> Dict[str, ReadCounts]:
    allele_counts = {}
    for seq, quality in zip(pc.get_query_sequences(bool_add_indels=True), pc.get_query_qualities()):
        if seq == '>' or seq == '<' or seq == '*':
            continue

        upper_seq = seq.upper()
        if is_forward_strand(seq):
            strand_direction = StrandDirection.FORWARD
            forward_reads_count = quality
            reverse_reads_count = 0
        else:
            strand_direction = StrandDirection.REVERSE
            forward_reads_count = 0
            reverse_reads_count = quality

        # todo - consider probability for other alleles
        if upper_seq in allele_counts:
            allele_counts[upper_seq].add_count(quality, strand_direction)
        else:
            allele_counts[upper_seq] = ReadCounts(forward_support=forward_reads_count,
                                                  reverse_support=reverse_reads_count)
    return allele_counts


def count_alleles_in_gvcf(variant: VariantRecord) -> Dict[str, ReadCounts]:
    """
    * Assume variant contains a single sample
    * Assume <NON_REF> allele is always last
    * ignore <NON_REF> allele since SB field ignored it
    * Count the number of reads supporting each allele/strand
    * If no variant is called on this sample, we don't know the strand bias
    """
    ref = variant.ref
    allele_counts = {}
    alts = get_filtered_alleles_list(variant)[1:]

    call = variant.samples[0]
    if call['GT'] == (None, None):
        allele_counts[ref] = ReadCounts()
        return allele_counts

    # most variants and some ref called contain AD and SB fields
    if 'SB' in call:
        is_multi_allelic = len(alts) > 1
        if is_multi_allelic:
            # For multi-allelic variant take per-strand counts from AS_SB_TABLE
            # re-split by '|' and ',' separators
            strand_bias = [int(sb) for sb in '|'.join(variant.info['AS_SB_TABLE']).split('|')]

        else:
            # For bi-allelic variant take per-strand counts from SB field
            strand_bias = call['SB']

        allele_counts[ref] = ReadCounts(forward_support=strand_bias[0], reverse_support=strand_bias[1])
        for i, alt in enumerate(alts):
            allele_counts[alt] = ReadCounts(forward_support=int(strand_bias[2 + 2 * i]),
                                            reverse_support=int(strand_bias[3 + 2 * i]))
    else:
        allele_counts[ref] = ReadCounts(unknown_strand_support=int(call['DP']))
    return allele_counts