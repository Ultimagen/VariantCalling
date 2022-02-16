import unittest

from ugvc.pipelines.sec.correct_systematic_errors import called_excluded_alleles


class TestAlleleExclusion(unittest.TestCase):

    def test_excluded_alleles(self):
        self.assertTrue(called_excluded_alleles(excluded_refs=['T'],
                                                all_excluded_alts=[['A']],
                                                called_ref='T',
                                                called_alts={'A'}))

    def test_non_excluded_alt(self):
        self.assertFalse(called_excluded_alleles(excluded_refs=['T'],
                                                 all_excluded_alts=[['A']],
                                                 called_ref='T',
                                                 called_alts={'TA'}))

        self.assertTrue(called_excluded_alleles(excluded_refs=['T'],
                                                all_excluded_alts=[['A']],
                                                called_ref='T',
                                                called_alts={'A'}))

    def test_non_excluded_ref(self):
        self.assertFalse(called_excluded_alleles(excluded_refs=['TAA'],
                                                 all_excluded_alts=[['T']],
                                                 called_ref='TA',
                                                 called_alts={'T'}))

        self.assertTrue(called_excluded_alleles(excluded_refs=['TAA'],
                                                all_excluded_alts=[['T']],
                                                called_ref='TAA',
                                                called_alts={'T'}))

    def test_non_excluded_alt_one_out_of_two_alts(self):
        self.assertFalse(called_excluded_alleles(excluded_refs=['TAA'],
                                                 all_excluded_alts=[['T']],
                                                 called_ref='TAA',
                                                 called_alts={'T', 'TA'}))

        self.assertFalse(called_excluded_alleles(excluded_refs=['TAA'],
                                                 all_excluded_alts=[['T']],
                                                 called_ref='TAA',
                                                 called_alts={'TA', 'TAAAA'}))

    def test_non_excluded_ref_two_options(self):
        self.assertTrue(called_excluded_alleles(excluded_refs=['TAA', 'TAAA'],
                                                all_excluded_alts=[['T'], ['T']],
                                                called_ref='TAAA',
                                                called_alts={'T'}))
        self.assertFalse(called_excluded_alleles(excluded_refs=['TAA', 'TAAA'],
                                                 all_excluded_alts=[['T'], ['T']],
                                                 called_ref='TA',
                                                 called_alts={'T'}))

    def test_non_excluded_alt_two_options(self):
        self.assertTrue(called_excluded_alleles(excluded_refs=['T', 'T'],
                                                all_excluded_alts=[['TA'], ['TAA']],
                                                called_ref='T',
                                                called_alts={'TAA'}))
        self.assertFalse(called_excluded_alleles(excluded_refs=['T', 'T'],
                                                 all_excluded_alts=[['TA'], ['TAA']],
                                                 called_ref='T',
                                                 called_alts={'TAAA'}))
