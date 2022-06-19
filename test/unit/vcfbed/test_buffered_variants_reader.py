import unittest
from test import test_dir

from ugvc.vcfbed.buffered_variant_reader import BufferedVariantReader


class TestBufferedVariantReader(unittest.TestCase):
    def test_get_variant(self):
        reader = BufferedVariantReader(f"{test_dir}/resources/single_sample_example.vcf.gz")
        variant_1 = reader.get_variant("chr1", 930196)
        self.assertEqual(("T", "TG", "<NON_REF>"), variant_1.alleles)
        variant_2 = reader.get_variant("chr1", 1044019)
        self.assertEqual(("G", "GC", "<NON_REF>"), variant_2.alleles)
        variant_3 = reader.get_variant("chr1", 10)
        self.assertIsNone(variant_3)

    def test_header(self):
        reader = BufferedVariantReader(f"{test_dir}/resources/single_sample_example.vcf.gz")
        self.assertEqual("HG00239", reader.header.samples[0])
