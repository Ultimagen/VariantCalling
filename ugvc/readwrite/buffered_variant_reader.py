from typing import Optional

import pysam
from pysam import VariantRecord


class BufferedVariantReader:

    def __init__(self, file_name: str):
        self.pysam_reader = pysam.VariantFile(file_name)
        self.header = self.pysam_reader.header
        self.variants = {}
        self.current_chromosome = ''

    def get_variant(self, chromosome: str, pos: int) -> Optional[VariantRecord]:
        if chromosome == self.current_chromosome:
            if pos in self.variants:
                return self.variants[pos]
        self.current_chromosome = chromosome
        self.variants = {}

        for variant in self.pysam_reader.fetch(chromosome, pos - 1, pos + 1000):
            self.variants[variant.pos] = variant

        if pos in self.variants:
            return self.variants[pos]
        else:
            return None
