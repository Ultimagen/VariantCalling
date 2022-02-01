import unittest

from ugvc.dna.strand_direction import forward_strand, reverse_strand
from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.read_counts import ReadCounts

ref_gt = '0/0'


class TestConditionalAlleleDistribution(unittest.TestCase):

    def test_construction(self):
        cad = ConditionalAlleleDistribution(conditioned_alleles='A,T',
                                            conditioned_genotype=ref_gt,
                                            observed_alleles='A,AG',
                                            allele_counts={'A': ReadCounts(10, 8), 'AG': ReadCounts(1, 0)})

        self.assertEqual({'0/0': {'A,AG': 1}}, cad.num_of_samples_with_alleles)
        self.assertEqual({'A,AG': {'A': ReadCounts(10, 8), 'AG': ReadCounts(1, 0)}},
                         cad.allele_counts_dict[ref_gt])
        self.assertEqual(1, cad.get_observed_alleles_frequency(ref_gt, 'A,AG'))
        self.assertEqual(10, cad.get_allele_count(ref_gt, 'A,AG', 'A', forward_strand))
        self.assertEqual(8, cad.get_allele_count(ref_gt, 'A,AG', 'A', reverse_strand))
        self.assertEqual(1, cad.get_allele_count(ref_gt, 'A,AG', 'AG', forward_strand))
        self.assertEqual(0, cad.get_allele_count(ref_gt, 'A,AG', 'AG', reverse_strand))
        self.assertEqual('A 10,8,0 AG 1,0,0', cad.get_allele_counts_string(ref_gt, 'A,AG'))

        # does not return 0 for unobserved alleles
        self.assertEqual(0.01, cad.get_observed_alleles_frequency(ref_gt, 'A,T'))

    def test_merging_two_records_with_same_alleles(self):
        cad = ConditionalAlleleDistribution('A,T', ref_gt, 'A,T', {'A': ReadCounts(10, 8), 'T': ReadCounts(1, 0)})
        cad.update_distribution(cad)
        self.assertEqual(ReadCounts(20, 16), cad.get_allele_counts(ref_gt, 'A,T', 'A'))
        self.assertEqual(ReadCounts(2, 0), cad.get_allele_counts(ref_gt, 'A,T', 'T'))

    def test_merging_two_records_with_different_alleles(self):
        cad1 = ConditionalAlleleDistribution('A,T', ref_gt, 'A,T', {'A': ReadCounts(10, 8), 'T': ReadCounts(1, 0)})
        cad2 = ConditionalAlleleDistribution('A,T', ref_gt, 'A,T', {'A': ReadCounts(8, 12), 'G': ReadCounts(0, 1)})
        cad1.update_distribution(cad2)
        self.assertEqual(ReadCounts(18, 20), cad1.get_allele_counts(ref_gt, 'A,T', 'A'))
        self.assertEqual(ReadCounts(1, 0), cad1.get_allele_counts(ref_gt, 'A,T', 'T'))
        self.assertEqual(ReadCounts(0, 1), cad1.get_allele_counts(ref_gt, 'A,T', 'G'))

    def test_merging_two_record_with_unknown_strand_direction(self):
        cad1 = ConditionalAlleleDistribution('A,T', ref_gt, 'A,T', {'A': ReadCounts(10, 8), 'T': ReadCounts(1)})
        cad2 = ConditionalAlleleDistribution('A,T', ref_gt, 'A,T', {'A': ReadCounts(0, 0, 20)})
        cad1.update_distribution(cad2)
        self.assertEqual(ReadCounts(10, 8, 20), cad1.get_allele_counts(ref_gt, 'A,T', 'A'))
        self.assertEqual(ReadCounts(1, 0), cad1.get_allele_counts(ref_gt, 'A,T', 'T'))

    def test_merging_two_records_with_different_genotype(self):
        """
        Merge does not work on mismatching genotypes
        """
        cad1 = ConditionalAlleleDistribution('A,T', '0/0', 'A,T', {'A': ReadCounts(10, 8), 'T': ReadCounts(1, 0)})
        cad2 = ConditionalAlleleDistribution('A,T', '1/1', 'A,T', {'A': ReadCounts(), 'T': ReadCounts(11, 12)})
        cad1.update_distribution(cad2)
        self.assertEqual(ReadCounts(10, 8), cad1.get_allele_counts('0/0', 'A,T', 'A'))
        self.assertEqual(ReadCounts(11, 12), cad1.get_allele_counts('1/1', 'A,T', 'T'))
