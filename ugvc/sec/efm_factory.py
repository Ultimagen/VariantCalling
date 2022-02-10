from typing import Tuple

import numpy as np
from pysam import PileupColumn

from dna.strand_direction import StrandDirection
from ugvc.sec.allele_counter import count_alleles_in_pileup
from ugvc.sec.error_frequency_matrix import ErrorFrequencyMatrix


def pileup_to_efm(pc: PileupColumn, true_genotype: Tuple[str, str]) -> ErrorFrequencyMatrix:
    matrix = np.zeros((6, 2))
    allele_counts = count_alleles_in_pileup(pc)
    for allele, read_counts in allele_counts.items():
        error_type = get_error_type(allele, true_genotype)
        matrix[error_type, 0] += read_counts.get_count(StrandDirection.REVERSE)
        matrix[error_type, 1] += read_counts.get_count(StrandDirection.FORWARD)
    return ErrorFrequencyMatrix(matrix)


def get_error_type(allele: str, genotype: Tuple[str, str]):
    if allele == genotype[0] or allele == genotype[1]:
        return 1
    if allele == '-1N':
        return 0
    # homopolymer insertion of 1 base
    if len(allele) == 2 and allele[0] == allele[1] and (genotype[0][-1] == allele[0] or genotype[1][-1] == allele[0]):
        return 2
    # homopolymer insertion of 2 bases
    if len(allele) == 3 and allele[0] == allele[1] == allele[2] and (
            genotype[0][-1] == allele[0] or genotype[1][-1] == allele[0]):
        return 3
    # base error mutation
    if len(allele) == 1:
        return 4
    # other error
    return 5
