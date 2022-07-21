from ugvc.cnv.filter_sample_cnvs import annotate_bed

import unittest
from os.path import join as pjoin
import filecmp

from test import get_resource_dir
inputs_dir = get_resource_dir(__file__)


class TestFilterSampleCnvs(unittest.TestCase):
    def test_annotate_bed(self):
        input_bed_file = pjoin(inputs_dir, "unfiltered_cnvs.bed")
        expected_out_filtered_bed_file = pjoin(inputs_dir, "filtered_cnvs.bed")
        coverage_lcr_file = pjoin(inputs_dir, "UG-CNV-LCR.bed")
        blocklist = pjoin(inputs_dir, "blocklist.bed")
        intersection_cutoff = 0.5
        min_cnv_length = 10000
        [out_annotate_file, out_filtered_file] = annotate_bed(input_bed_file, intersection_cutoff,
                                                              coverage_lcr_file, intersection_cutoff,
                                                              blocklist, min_cnv_length)

        self.assertTrue(filecmp.cmp(out_filtered_file, expected_out_filtered_bed_file),
                        'filtered cnvs list is not identical to expected list')

