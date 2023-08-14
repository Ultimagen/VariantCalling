import filecmp
import os
from os.path import join as pjoin
from os.path import dirname

from test import get_resource_dir, test_dir

from ugvc.somatic_cnv import bicseq2_post_processing

class TestBicseq2PostProcessing:
    inputs_dir = get_resource_dir(__file__)

    def test_bicseq2_post_processing(self, tmpdir):
        input_bicseq2_txt_file = pjoin(self.inputs_dir, "T_N_HCC1143_CHR22_test.bicseq2.txt")
        expected_out_bed_file = pjoin(self.inputs_dir, "expected_T_N_HCC1143_CHR22_test.bicseq2.bed")
        prefix = f"{tmpdir}/"
        out_file = f"{tmpdir}/T_N_HCC1143_CHR22_test.bicseq2.bed"

        bicseq2_post_processing.run([
            "bicseq2_post_processing",
             "--input_bicseq2_txt_file",
             input_bicseq2_txt_file,
             "--out_directory",
             prefix
            ])
        assert filecmp.cmp(expected_out_bed_file, out_file)
