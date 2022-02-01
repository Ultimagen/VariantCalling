from ugvc.dna.strand_direction import StrandDirection, forward_strand, reverse_strand, unknown_strand_direction


class ReadCounts:

    def __init__(self, forward_support: int = 0, reverse_support: int = 0, unknown_strand_support: int = 0):
        self.__forward_support = forward_support
        self.__reverse_support = reverse_support
        self.__unknown_strand_support = unknown_strand_support

    def add_count(self, num_of_reads, strand: StrandDirection):
        if strand.forward:
            self.__forward_support += num_of_reads
        elif strand.reverse:
            self.__reverse_support += num_of_reads
        else:
            self.__unknown_strand_support += num_of_reads

    def add_counts(self, read_counts):
        self.__forward_support += read_counts.get_count(forward_strand)
        self.__reverse_support += read_counts.get_count(reverse_strand)
        self.__unknown_strand_support += read_counts.get_count(unknown_strand_direction)

    def get_count(self, strand: StrandDirection) -> int:
        if strand.forward:
            return self.__forward_support
        elif strand.reverse:
            return self.__reverse_support
        else:
            return self.__unknown_strand_support

    def get_counts_tuple_as_str(self) -> str:
        return f'{self.__forward_support},{self.__reverse_support},{self.__unknown_strand_support}'

    def __repr__(self):
        if (self.__forward_support > 0 or self.__reverse_support > 0) and self.__unknown_strand_support == 0:
            return f'{self.__forward_support}(+) {self.__reverse_support}(-)'
        if (self.__forward_support == 0 or self.__reverse_support == 0) and self.__unknown_strand_support > 0:
            return f'{self.__unknown_strand_support}(?)'
        if self.__forward_support == 0 and self.__reverse_support == 0 and self.__unknown_strand_support == 0:
            return '0'
        else:
            return f'{self.__forward_support}(+) {self.__reverse_support}(-) {self.__unknown_strand_support}(?)'

    def __eq__(self, other):
        return self.__forward_support == other.__forward_support and \
               self.__reverse_support == other.__reverse_support and \
               self.__unknown_strand_support == other.__unknown_strand_support
