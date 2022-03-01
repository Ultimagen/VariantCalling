import pickle
from typing import Dict, Tuple, List

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution


class ConditionalAlleleDistributions:

    """
    chromosome -> position -> conditioned_genotype -> ConditionalAlleleDistribution
    """
    def __init__(self, pickle_files: List[str] = None):
        """
        Construct a new, or existing (from pickles_prefix) ConditionalAlleleDistributions object
        """
        self.distributions_per_chromosome: Dict[str, Dict[int, ConditionalAlleleDistribution]] = {}

        if pickle_files is not None:
            for pickle_file in pickle_files:
                chr_name = pickle_file.split('.')[-2]
                with open(pickle_file, 'rb') as fh:
                    self.distributions_per_chromosome[chr_name] = pickle.load(fh)

    def add_counts(self, chrom: str, pos: int, conditional_allele_distribution: ConditionalAlleleDistribution):
        if chrom not in self.distributions_per_chromosome:
            self.distributions_per_chromosome[chrom] = {}
        dist_per_chrom = self.distributions_per_chromosome[chrom]
        if pos not in dist_per_chrom:
            dist_per_chrom[pos] = conditional_allele_distribution
        else:
            dist_per_chrom[pos].update_distribution(conditional_allele_distribution)

    def get_distributions_per_locus(self, chrom: str, pos: int) -> ConditionalAlleleDistribution:
        return self.distributions_per_chromosome[chrom][pos]

    def __iter__(self) -> Tuple[str, int, ConditionalAlleleDistribution]:
        for chrom, distributions_per_pos in self.distributions_per_chromosome.items():
            for pos, cad in sorted(distributions_per_pos.items()):
                    return chrom, pos, cad
