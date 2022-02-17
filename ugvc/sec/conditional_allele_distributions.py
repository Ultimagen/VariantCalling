from typing import Dict, Tuple

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution


class ConditionalAlleleDistributions:
    """
    chromosome -> position -> conditioned_genotype -> ConditionalAlleleDistribution
    """
    def __init__(self):
        self.distributions_per_chromosome: Dict[str, Dict[int, ConditionalAlleleDistribution]] = {}

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
