import unittest

from ugvc.sec.systematic_error_correction_call import SECCallType
from ugvc.sec.systematic_error_correction_caller import SECCaller
from test.unit.sec.test_sets import KnownHetIns, NoVariantWithNoise, UncorrelatedSnp, NoReferenceGenotype, \
    HomVarWithTwoEquivalentHetGenotypes


class TestSecCaller(unittest.TestCase):
    sec_caller = SECCaller(0.0001, 0.001, 0.99, novel_detection_only=False, replace_to_known_genotype=False)

    def test_reject_novel_variant_observed_as_noise(self):
        ts = NoVariantWithNoise()
        ts.observed_variant.alleles = ['C', 'CG']
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [0, 1]
        sample_info['SB'] = (12, 10, 0, 2)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.reference, sec_call.call_type)
        self.assertEqual('C,CG', sec_call.alleles)
        self.assertEqual('0/0', sec_call.genotype)
        self.assertAlmostEqual(0.7635, sec_call.novel_variant_p_value, places=3)
        self.assertEqual(None, sec_call.genotype_quality)
        self.assertEqual(None, sec_call.gt_correlation)

        sec_caller_with_replace = SECCaller(0.0001, 0.001, 0.99, novel_detection_only=False,
                                            replace_to_known_genotype=True)
        sec_call = sec_caller_with_replace.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.reference, sec_call.call_type)
        self.assertEqual('C', sec_call.alleles)  # notice how alleles are currently taken from ground-truth
        self.assertEqual('0/0', sec_call.genotype)

    def test_call_known_variant(self):
        ts = KnownHetIns()
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [0, 1]
        sample_info['SB'] = (12, 10, 8, 9)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        [print(r) for r in sec_call.sec_records]
        self.assertEqual(SECCallType.known, sec_call.call_type)
        self.assertEqual('A,AG', sec_call.alleles)  # notice how alleles are currently taken from ground-truth
        self.assertEqual('0/1', sec_call.genotype)
        self.assertAlmostEqual(0.4893, sec_call.novel_variant_p_value, places=3)
        self.assertEqual(140, sec_call.genotype_quality)
        self.assertEqual(1, sec_call.gt_correlation)

    def test_call_novel_variant_even_if_observed_as_noise(self):
        ts = NoVariantWithNoise()
        ts.observed_variant.alleles = ['C', 'CG']
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [0, 1]
        sample_info['SB'] = (12, 10, 8, 13)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        for r in sec_call.sec_records:
            print(r)
        self.assertEqual(SECCallType.novel, sec_call.call_type)
        self.assertEqual('C,CG', sec_call.alleles)  # notice how alleles are currently taken from ground-truth
        self.assertEqual('0/1', sec_call.genotype)
        self.assertAlmostEqual(9.9 * 10 ** -9, sec_call.novel_variant_p_value)
        self.assertEqual(None, sec_call.genotype_quality)
        self.assertEqual(None, sec_call.gt_correlation)

    def test_reject_unobserved_novel_variant_with_few_reads(self):
        ts = NoVariantWithNoise()
        ts.observed_variant.alleles = ['C', 'CGG']
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [0, 1]
        sample_info['SB'] = (12, 10, 2, 3)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.reference, sec_call.call_type)
        self.assertEqual('C,CGG', sec_call.alleles)
        self.assertEqual('0/0', sec_call.genotype)
        self.assertAlmostEqual(0.0338, sec_call.novel_variant_p_value, places=3)
        self.assertEqual(None, sec_call.genotype_quality)
        self.assertEqual(None, sec_call.gt_correlation)

        sec_caller_with_replace = SECCaller(0.0001, 0.001, 0.99, novel_detection_only=False,
                                            replace_to_known_genotype=True)
        sec_call = sec_caller_with_replace.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.reference, sec_call.call_type)
        self.assertEqual('C', sec_call.alleles)
        self.assertEqual('0/0', sec_call.genotype)

    def test_detect_uncorrelated_variant(self):
        ts = UncorrelatedSnp()
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [0, 1]
        sample_info['SB'] = (12, 10, 5, 5)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.uncorrelated, sec_call.call_type)
        self.assertEqual('A,T', sec_call.alleles)  # notice how alleles are currently taken from ground-truth
        self.assertEqual('0/1', sec_call.genotype)
        self.assertAlmostEqual(0.3643, sec_call.novel_variant_p_value, places=3)
        self.assertEqual(None, sec_call.genotype_quality)
        self.assertAlmostEqual(0.0946, sec_call.gt_correlation, places=3)

    def test_variant_without_ground_truth_ref_genotype(self):
        ts = NoReferenceGenotype()
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [0, 1]
        sample_info['SB'] = (16, 12, 14, 12)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.unobserved, sec_call.call_type)
        self.assertEqual('A,G', sec_call.alleles)
        self.assertEqual('0/1', sec_call.genotype)
        self.assertAlmostEqual(0.90164, sec_call.novel_variant_p_value, places=3)
        self.assertEqual(730, sec_call.genotype_quality)
        self.assertAlmostEqual(1, sec_call.gt_correlation, places=3)

    def test_call_homozygous_variant_with_to_equivalent_unlikely_options(self):
        ts = HomVarWithTwoEquivalentHetGenotypes()
        sample_info = ts.observed_variant.samples[0]
        sample_info['GT'] = [1, 1]
        sample_info['SB'] = (0, 0, 10, 12)
        sec_call = self.sec_caller.call(ts.observed_variant, ts.expected_distribution)
        self.assertEqual(SECCallType.known, sec_call.call_type)
        self.assertEqual('A,T', sec_call.alleles)  # notice how alleles are currently taken from ground-truth
        self.assertEqual('1/1', sec_call.genotype)
        self.assertAlmostEqual(0.25137, sec_call.novel_variant_p_value, places=3)
        self.assertEqual(640, sec_call.genotype_quality)
        self.assertAlmostEqual(1, sec_call.gt_correlation, places=3)