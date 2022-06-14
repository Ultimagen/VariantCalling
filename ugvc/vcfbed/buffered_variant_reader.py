from __future__ import annotations

import pysam
from pysam import VariantRecord


class BufferedVariantReader:
    """
    A wrapper around pysam VariantFile.
    To be we used when random access to relatively sparse set of variants is needed.

    Reads a vcf/gvcf (can be zipped) files, and enables to efficiently fetch a specific variant from locus.
    pysam.VariantFile fetch returns a block of variants on each fetch.
    Here we cache this block of variants for quicker access in the future.
    """

    # pylint:disable=too-few-public-methods
    def __init__(self, file_name: str):
        self.pysam_reader = pysam.VariantFile(file_name)
        self.header = self.pysam_reader.header
        self.variants = {}
        self.current_chromosome = ""

    def get_variant(self, chromosome: str, pos: int) -> VariantRecord | None:
        if chromosome == self.current_chromosome:
            if pos in self.variants:
                return self.variants[pos]
        self.current_chromosome = chromosome
        self.variants = {}

        for variant in self.pysam_reader.fetch(chromosome, pos - 1, pos + 1000):
            self.variants[variant.pos] = variant

        if pos in self.variants:
            return self.variants[pos]
        return None
