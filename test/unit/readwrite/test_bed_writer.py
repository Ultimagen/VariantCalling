import unittest

from ugvc.readwrite.bed_writer import BedWriter
from ugvc.readwrite.buffered_variant_reader import BufferedVariantReader
from test import test_dir


class TestBedWriter(unittest.TestCase):

    def test_write(self):
        file_name = f'{test_dir}/test_outputs/example.bed'
        writer = BedWriter(file_name)
        writer.write('chr2', 100, 102, 'hmer-indel')
        writer.write('chr3', 120, 123, 'snp')
        writer.close()

        lines = open(file_name, 'r').readlines()
        self.assertEqual('chr2\t100\t102\thmer-indel\n', lines[0])
        self.assertEqual('chr3\t120\t123\tsnp\n', lines[1])
