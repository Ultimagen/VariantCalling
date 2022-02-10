import unittest

from ugvc.variants.classify_seq_errors import ClassifySeqErrors


class TestClassifySeqErrors(unittest.TestCase):
    def test_is_insertion(self):
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'TT').is_insertion())

    def test_get_insert(self):
        self.assertEqual('T', ClassifySeqErrors('ATG', 'T', 'TT').get_insert())
        self.assertEqual('AT', ClassifySeqErrors('ATG', 'T', 'TAT').get_insert())
        self.assertEqual('G', ClassifySeqErrors('ATG', 'TG', 'TGG').get_insert())

    def test_get_deletion(self):
        self.assertEqual('G', ClassifySeqErrors('ATG', 'TG', 'T').get_deletion())
        self.assertEqual('GG', ClassifySeqErrors('ATG', 'TGG', 'T').get_deletion())

    def test_is_insertion_seq_error(self):
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'TT').is_seq_error())
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'TG').is_seq_error())
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'TTT').is_seq_error())
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'TGG').is_seq_error())
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'TTGG').is_seq_error())

        self.assertFalse(ClassifySeqErrors('ATG', 'T', 'TGA').is_seq_error())
        self.assertFalse(ClassifySeqErrors('ATG', 'T', 'TGGA').is_seq_error())

    def test_is_deletion_seq_error(self):
        self.assertTrue(ClassifySeqErrors('ATG', 'TG', 'T').is_seq_error())
        self.assertTrue(ClassifySeqErrors('ATG', 'TGG', 'T').is_seq_error())

        self.assertFalse(ClassifySeqErrors('ATG', 'TGA', 'T').is_seq_error())
        self.assertFalse(ClassifySeqErrors('ATG', 'TGGA', 'T').is_seq_error())

    def test_is_snp_seq_error(self):
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'A').is_seq_error())
        self.assertTrue(ClassifySeqErrors('ATG', 'T', 'G').is_seq_error())

        self.assertFalse(ClassifySeqErrors('ATG', 'T', 'C').is_seq_error())
        self.assertFalse(ClassifySeqErrors('GTG', 'T', 'A').is_seq_error())
