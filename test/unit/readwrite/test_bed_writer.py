import os
import unittest
from os.path import dirname

from test import test_dir
from ugvc.readwrite.bed_writer import BedWriter


class TestBedWriter(unittest.TestCase):

    def test_write(self):
        file_name = f'{test_dir}/test_outputs/readwrite/test_bed_writer/example.bed'
        os.makedirs(dirname(file_name), exist_ok=True)
        writer = BedWriter(file_name)
        writer.write('chr2', 100, 102, 'hmer-indel')
        writer.write('chr3', 120, 123, 'snp')
        writer.close()

        lines = open(file_name, 'r').readlines()
        self.assertEqual('chr2\t100\t102\thmer-indel\n', lines[0])
        self.assertEqual('chr3\t120\t123\tsnp\n', lines[1])
