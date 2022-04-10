import numpy as np

from ugvc.dna.strand_direction import StrandDirection


class ErrorFrequencyMatrix:
    """
    (chr1, pos1, A/A)  | count (+) | count (-)|
    -----------------------------------------
    | 0 (missed base) |  1         |   2     |
    | 1 (no_error)    | 980        |   964   |
    | 2               | 15         |   30    |
    | 3               | 1          |   5     |
    | base_mutation   | 1          |   1     |
    | other           | 0          |   0     |
    """

    base_mutation_index = 4
    other_error_index = 5

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def get_hmer_indel_rate(self, num_of_copies: int, strand: StrandDirection) -> float:
        num_of_copies = max(num_of_copies, 3)
        if strand.unknown:
            return self.matrix[num_of_copies, :].sum()
        column = 0 if strand.forward else 1
        return self.matrix[num_of_copies, column]

    def get_no_error_rate(self, strand: StrandDirection) -> float:
        return self.get_hmer_indel_rate(1, strand)

    def get_base_mutation_rate(self, strand: StrandDirection) -> float:
        if strand.unknown:
            return self.matrix[self.base_mutation_index, :].sum()
        column = 0 if strand.forward else 1
        return self.matrix[self.base_mutation_index, column]

    def get_other_error_rate(self, strand: StrandDirection) -> float:
        if strand.unknown:
            if strand.unknown:
                return self.matrix[self.other_error_index, :].sum()
        column = 0 if strand.forward else 1
        return self.matrix[self.other_error_index, column]

    def __str__(self):
        return f"ErrorFrequencyMatrix:\n{self.matrix}"
