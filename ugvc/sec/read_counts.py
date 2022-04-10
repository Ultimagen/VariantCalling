from typing import Tuple

from ugvc.dna.strand_direction import StrandDirection


class ReadCounts:
    def __init__(
        self,
        forward_support: int = 0,
        reverse_support: int = 0,
        unknown_strand_support: int = 0,
    ):
        self.__forward_support = forward_support
        self.__reverse_support = reverse_support
        self.__unknown_strand_support = unknown_strand_support

    def add_count(self, num_of_reads, strand: StrandDirection):
        if strand == StrandDirection.FORWARD:
            self.__forward_support += num_of_reads
        elif strand.REVERSE:
            self.__reverse_support += num_of_reads
        else:
            self.__unknown_strand_support += num_of_reads

    def add_counts(self, read_counts):
        self.__forward_support += read_counts.get_count(StrandDirection.FORWARD)
        self.__reverse_support += read_counts.get_count(StrandDirection.REVERSE)
        self.__unknown_strand_support += read_counts.get_count(StrandDirection.UNKNOWN)

    def get_counts(self) -> Tuple[int, int, int]:
        return (
            self.__forward_support,
            self.__reverse_support,
            self.__unknown_strand_support,
        )

    def get_count(self, strand: StrandDirection) -> int:
        if strand == StrandDirection.FORWARD:
            return self.__forward_support
        elif strand == StrandDirection.REVERSE:
            return self.__reverse_support
        else:
            return self.__unknown_strand_support

    def get_counts_tuple_as_str(self) -> str:
        return f"{self.__forward_support},{self.__reverse_support},{self.__unknown_strand_support}"

    def __repr__(self):
        if (
            self.__forward_support > 0 or self.__reverse_support > 0
        ) and self.__unknown_strand_support == 0:
            return f"{self.__forward_support}(+) {self.__reverse_support}(-)"
        if (
            self.__forward_support == 0 or self.__reverse_support == 0
        ) and self.__unknown_strand_support > 0:
            return f"{self.__unknown_strand_support}(?)"
        if (
            self.__forward_support == 0
            and self.__reverse_support == 0
            and self.__unknown_strand_support == 0
        ):
            return "0"
        else:
            return f"{self.__forward_support}(+) {self.__reverse_support}(-) {self.__unknown_strand_support}(?)"

    def __eq__(self, other):
        return (
            self.__forward_support == other.__forward_support
            and self.__reverse_support == other.__reverse_support
            and self.__unknown_strand_support == other.__unknown_strand_support
        )
