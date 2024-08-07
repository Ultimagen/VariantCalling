import filecmp
from os.path import join as pjoin
import warnings
warnings.filterwarnings('ignore')
from test import get_resource_dir, test_dir

from ugvc.cnv import CNV_eval

class TestCnvEval:
    inputs_dir = get_resource_dir(__file__)

    def test_cnv_eval(self, tmpdir):
        sample1_bed_file = pjoin(self.inputs_dir, "COLO829.cnvs.annotate.bed")
        sample2_bed_file = pjoin(self.inputs_dir, "COLO-829--COLO-829BL.GT.bed")
        sample1_basename = "COLO829"
        sample2_basename = "GT"
        ug_cnv_lcr = pjoin(self.inputs_dir, "ug_cnv_lcr.v2.1.1.bed")
        
        expected_out_concordance = pjoin(self.inputs_dir,"expected_COLO829_GT.concordance.csv")
        
        out_dir = f"{tmpdir}"

        CNV_eval.run([
            "CNV_eval",
            "--sample1_bed_file",
            sample1_bed_file,
            "--sample2_bed_file",
            sample2_bed_file,
            "--sample1_basename",
            sample1_basename,
            "--sample2_basename",
            sample2_basename,
            "--out_dir",
            out_dir,
            "--ug_cnv_lcr",
            ug_cnv_lcr
            ])
        
        out_concordance = pjoin(tmpdir,f"{sample1_basename}_{sample2_basename}",f"{sample1_basename}_{sample2_basename}.concordance.csv")
        assert filecmp.cmp(out_concordance, expected_out_concordance)