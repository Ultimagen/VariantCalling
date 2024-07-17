import filecmp
import os
from os.path import join as pjoin
from os.path import dirname
import hashlib
import warnings
warnings.filterwarnings('ignore')

from test import get_resource_dir, test_dir
from ugvc.pipelines.cnv import annotate_FREEC_segments

class TestAnnotateFREECSegments:
    inputs_dir = get_resource_dir(__file__)

    def test_annotate_FREEC_segments(self, tmpdir):
        input_segments_file = pjoin(self.inputs_dir, "in_segments.txt")
        gain_cutoff = 1.03
        loss_cutoff = 0.97
        expected_out_segments_annotated = pjoin(self.inputs_dir,'expected_in_segments_annotated.txt')
        expected_out_segments_CNVs = pjoin(self.inputs_dir,'expected_in_segments_CNVs.bed')
        
        annotate_FREEC_segments.run([
            "annotate_FREEC_segments",
            "--input_segments_file",
            input_segments_file,
            "--gain_cutoff",
            str(gain_cutoff),
            "--loss_cutoff",
            str(loss_cutoff),
            ])
        
        out_segments_annotated = os.path.basename(input_segments_file) + '_annotated.txt'
        out_segments_CNVs = os.path.basename(input_segments_file) + '_CNVs.bed'
        assert filecmp.cmp(out_segments_annotated, expected_out_segments_annotated)
        assert filecmp.cmp(out_segments_CNVs, expected_out_segments_CNVs)
        
        