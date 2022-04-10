import unittest

from ugvc.vcfbed.genotype import Genotype


class TestGenotype(unittest.TestCase):
    def test_convert_to_reference(self):
        self.assertEqual("0/0", Genotype("0/1").convert_to_reference())
        self.assertEqual("0/0", Genotype("0/0").convert_to_reference())
        self.assertEqual("0/0", Genotype("1/1").convert_to_reference())
        self.assertEqual("0", Genotype("1").convert_to_reference())
        self.assertEqual("0", Genotype("0").convert_to_reference())

    def test_is_reference(self):
        self.assertFalse(Genotype("0/1").is_reference())
        self.assertTrue(Genotype("0/0").is_reference())
        self.assertFalse(Genotype("1/1").is_reference())
        self.assertFalse(Genotype("1").is_reference())
        self.assertTrue(Genotype("0").is_reference())
