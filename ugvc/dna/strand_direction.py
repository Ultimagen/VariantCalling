class StrandDirection:

    def __init__(self, direction: int):
        self.forward = False
        self.reverse = False
        self.unknown = False
        if direction == 1:
            self.forward = True
        elif direction == 0:
            self.reverse = True
        else:
            self.unknown = True


forward_strand = StrandDirection(1)
reverse_strand = StrandDirection(0)
unknown_strand_direction = StrandDirection(-1)


def is_forward_strand(allele: str):
    """
    pileup alleles with uppercase letters represent forward strand
    todo - handle indels
    """
    return any([x in allele for x in ['A', 'C', 'G', 'T']])
