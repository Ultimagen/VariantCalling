import unittest

import pysam

from test import test_dir
from ugvc.utils.pysam_utils import *


class TestPysamUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.vcf = pysam.VariantFile(f'{test_dir}/resources/single_sample_example.vcf')
        self.variant = next(self.vcf)

    def test_get_alleles_str(self):
        self.assertEqual('T,TG,<NON_REF>', get_alleles_str(self.variant))

    def test_get_filtered_alleles_list(self):
        self.assertEqual(['T','TG'], get_filtered_alleles_list(self.variant))
        self.assertEqual(['T'], get_filtered_alleles_list(self.variant,filter_list=['TG', '<NON_REF>']))

        # '*' as minor allele is filtered out automatically
        self.variant.alts = ('TG', '*')
        self.assertEqual(['T', 'TG'], get_filtered_alleles_list(self.variant))

        # '*' as major allele is not filtered out
        self.variant.alts = ('*', 'TG')
        self.assertEqual(['T', '*', 'TG'], get_filtered_alleles_list(self.variant))

        # '*' as major allele can be filtered out explicitly
        self.variant.alts = ('*', 'TG')
        self.assertEqual(['T', 'TG'], get_filtered_alleles_list(self.variant, filter_list=['*']))

    def test_get_filtered_alleles_str(self):
        self.assertEqual('T,TG', get_filtered_alleles_str(self.variant))
        self.assertEqual('T', get_filtered_alleles_str(self.variant, filter_list=['TG', '<NON_REF>']))

        # '*' as minor allele is filtered out automatically
        self.variant.alts = ('TG', '*')
        self.assertEqual('T,TG', get_filtered_alleles_str(self.variant))

        # '*' as major allele is not filtered out
        self.variant.alts = ('*', 'TG')
        self.assertEqual('T,*,TG', get_filtered_alleles_str(self.variant))

        # '*' as major allele can be filtered out explicitly
        self.variant.alts = ('*', 'TG')
        self.assertEqual('T,TG', get_filtered_alleles_str(self.variant, filter_list=['*']))


    def test_get_genotype(self):
        self.assertEqual('T/T', get_genotype(self.variant.samples[0]))

    def test_get_genotype_indices(self):
        self.assertEqual('0/0', get_genotype_indices(self.variant.samples[0]))

    def test_has_candidate_alternatives(self):
        self.assertEqual(('T', 'TG', '<NON_REF>'), self.variant.alleles)
        self.assertTrue(has_candidate_alternatives(self.variant))

        # scroll to first variant without alternative
        variant = None
        for i in range(7):
            variant = next(self.vcf)
        self.assertEqual(('C', '<NON_REF>'), variant.alleles)
        self.assertFalse(has_candidate_alternatives(variant))

    def test_is_snp(self):
        self.assertTrue(is_snp(['A', 'T']))
        self.assertFalse(is_snp(['A', 'AG']))
        self.assertTrue(is_snp(['A', 'T', 'C']))
        self.assertFalse(is_snp(['A', 'T', 'AG']))
        self.assertFalse(is_snp(['AT', 'GC']))
