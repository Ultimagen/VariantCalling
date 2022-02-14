import unittest

from ugvc.sec.read_counts import ReadCounts
from ugvc.sec.systematic_error_correction_record import SECRecord
from test.unit.sec.variant_test_examples import KnownHetIns


class TestSecRecord(unittest.TestCase):
    test_set = KnownHetIns()

    def test_reference_matching_data(self):
        ts = self.test_set

        sec_record = SECRecord(chrom=ts.chrom,
                               pos=ts.pos,
                               expected_distribution=ts.expected_distribution,
                               conditioned_genotype='0/0',
                               observed_alleles='A,AG',
                               actual_allele_counts={'A': ReadCounts(12, 10),
                                                     'AG': ReadCounts(0, 2)})  # exactly the expected amount of noise
        sec_record.process()

        self.assertAlmostEqual(1, sec_record.likelihood_ratio)
        self.assertAlmostEqual(0.018, sec_record.likelihood, places=3)
        self.assertAlmostEqual(1.0, sec_record.forward_enrichment_pval, places=3)
        self.assertAlmostEqual(0.763, sec_record.reverse_enrichment_pval, places=3)
        self.assertAlmostEqual(0.763, sec_record.strand_enrichment_pval, places=3)
        self.assertAlmostEqual(0.763, sec_record.frequency_scaled_strand_enrichment_pval, places=3)

    def test_expected_observed_alleles_but_unexpected_distribution(self):
        ts = self.test_set

        sec_record = SECRecord(chrom=ts.chrom,
                               pos=ts.pos,
                               expected_distribution=ts.expected_distribution,
                               conditioned_genotype='0/0',
                               observed_alleles='A,AG',
                               actual_allele_counts={'A': ReadCounts(12, 10),
                                                     'AG': ReadCounts(10, 7)})  # unexpected amount of allele
        sec_record.process()

        self.assertAlmostEqual(7 * 10 ** -7, sec_record.likelihood_ratio, places=6)
        self.assertAlmostEqual(0, sec_record.likelihood, places=3)
        # reverse strand is more expected than forward
        self.assertAlmostEqual(0, sec_record.forward_enrichment_pval, places=3)
        self.assertAlmostEqual(0.0527, sec_record.reverse_enrichment_pval, places=3)
        self.assertAlmostEqual(0, sec_record.strand_enrichment_pval, places=3)
        self.assertAlmostEqual(0, sec_record.frequency_scaled_strand_enrichment_pval, places=3)

    def test_unexpected_observed_alleles(self):
        ts = self.test_set

        sec_record = SECRecord(chrom=ts.chrom,
                               pos=ts.pos,
                               expected_distribution=ts.expected_distribution,
                               conditioned_genotype='0/0',
                               observed_alleles='A,AGG',
                               actual_allele_counts={'A': ReadCounts(12, 10),
                                                     'AGG': ReadCounts(2, 3)})
        sec_record.process()
        self.assertAlmostEqual(0.0027, sec_record.likelihood_ratio, places=4)
        self.assertAlmostEqual(0, sec_record.likelihood, places=3)
        self.assertAlmostEqual(0.347, sec_record.forward_enrichment_pval, places=3)
        self.assertAlmostEqual(0.097, sec_record.reverse_enrichment_pval, places=3)
        self.assertAlmostEqual(0.0337, sec_record.strand_enrichment_pval, places=3)
        self.assertAlmostEqual(0, sec_record.frequency_scaled_strand_enrichment_pval, places=3)
