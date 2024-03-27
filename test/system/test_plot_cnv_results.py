import filecmp
import os
from os.path import join as pjoin
from os.path import dirname
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from test import get_resource_dir, test_dir

from ugvc.cnv import plot_cnv_results

class TestPlotCnvResults:
    inputs_dir = get_resource_dir(__file__)

    def test_plot_germline_cnv_results(self, tmpdir):
        input_germline_coverage = pjoin(self.inputs_dir, "NA11428.chr1_chr2.ReadCounts.bed")
        input_dup_bed = pjoin(self.inputs_dir, "NA11428.chr1_chr2.DUP.cnvs.filter.bed")
        input_del_bed = pjoin(self.inputs_dir, "NA11428.chr1_chr2.DEL.cnvs.filter.bed")

        expected_calls_fig = pjoin(self.inputs_dir,'expected_NA11428_cnv_figures/NA11428.chr1_chr2.CNV.calls.jpeg')
        expected_coverage_fig = pjoin(self.inputs_dir,'expected_NA11428_cnv_figures/NA11428.chr1_chr2.CNV.coverage.jpeg')
        expected_dup_del_fig =(self.inputs_dir,'expected_NA11428_cnv_figures/NA11428.chr1_chr2.dup_del.calls.jpeg')
        
        sample_name = 'NA11428.chr1_chr2'
        out_dir = f"{tmpdir}"
        plot_cnv_results.run([
            "plot_cnv_results",
            "--germline_coverage",
            input_germline_coverage,
            "--out_directory",
            out_dir,
            "--sample_name",
            sample_name,
            "--duplication_cnv_calls",
            input_dup_bed,
            "--deletion_cnv_calls",
            input_del_bed
            ])
        
        out_calls_fig = f"{tmpdir}/{sample_name}.CNV.calls.jpeg"
        out_coverage_fig = f"{tmpdir}/{sample_name}.CNV.coverage.jpeg"
        out_dup_del_fig = f"{tmpdir}/{sample_name}.dup_del.calls.jpeg"

        assert os.path.getsize(out_calls_fig)>0
        assert os.path.getsize(out_coverage_fig)>0
        assert os.path.getsize(out_dup_del_fig)>0
        #assert filecmp.cmp(expected_out_bed_file, out_file)
    
    
    def test_plot_somatic_cnv_results(self, tmpdir):
        input_germline_coverage = pjoin(self.inputs_dir, "NA11428.chr1_chr2.ReadCounts.bed")
        
        # input_dup_bed = pjoin(self.inputs_dir, "NA11428.chr1_chr2.DUP.cnvs.filter.bed")
        # input_del_bed = pjoin(self.inputs_dir, "NA11428.chr1_chr2.DEL.cnvs.filter.bed")

        expected_calls_fig = pjoin(self.inputs_dir,'expected_NA11428_cnv_figures/NA11428.chr1_chr2.CNV.calls.jpeg')
        expected_coverage_fig = pjoin(self.inputs_dir,'expected_NA11428_cnv_figures/NA11428.chr1_chr2.CNV.coverage.jpeg')
        expected_dup_del_fig =(self.inputs_dir,'expected_NA11428_cnv_figures/NA11428.chr1_chr2.dup_del.calls.jpeg')
        
        sample_name = 'NA11428.chr1_chr2'
        out_dir = f"{tmpdir}"
        plot_cnv_results.run([
            "plot_cnv_results",
            "--germline_coverage",
            input_germline_coverage,
            "--out_directory",
            out_dir,
            "--sample_name",
            sample_name,
            "--duplication_cnv_calls",
            input_dup_bed,
            "--deletion_cnv_calls",
            input_del_bed
            ])
        
        out_calls_fig = f"{tmpdir}/{sample_name}.CNV.calls.jpeg"
        out_coverage_fig = f"{tmpdir}/{sample_name}.CNV.coverage.jpeg"
        out_dup_del_fig = f"{tmpdir}/{sample_name}.dup_del.calls.jpeg"

        assert os.path.getsize(out_calls_fig)>0
        assert os.path.getsize(out_coverage_fig)>0
        assert os.path.getsize(out_dup_del_fig)>0
        #assert filecmp.cmp(expected_out_bed_file, out_file)
