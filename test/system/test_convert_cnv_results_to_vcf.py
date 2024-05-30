import filecmp
import os
from os.path import join as pjoin
from os.path import dirname
import hashlib
import warnings
warnings.filterwarnings('ignore')

from test import get_resource_dir, test_dir

from ugvc.cnv import convert_cnv_results_to_vcf


def compare_zipped_files(a,b):
    fileA = hashlib.sha256(open(a, 'rb').read()).digest()
    fileB = hashlib.sha256(open(b, 'rb').read()).digest()
    if fileA == fileB:
        return True
    else:
        return False


class TestConvertCnvResultsToVcf:
    inputs_dir = get_resource_dir(__file__)

    def test_convert_cnv_results_to_vcf(self, tmpdir):
        input_bed_file = pjoin(self.inputs_dir, "EL-0059.cnvs.annotate.bed")
        genome_file = pjoin(self.inputs_dir, "Homo_sapiens_assembly38.chr1-24.genome")
        expected_out_vcf = pjoin(self.inputs_dir,'EL-0059.cnv.vcf.gz')
        
        sample_name = 'EL-0059'
        out_dir = f"{tmpdir}"
        convert_cnv_results_to_vcf.run([
            "convert_cnv_results_to_vcf",
            "--cnv_annotated_bed_file",
            input_bed_file,
            "--out_directory",
            out_dir,
            "--sample_name",
            sample_name,
            "--fasta_index_file",
            genome_file
            ])
        
        out_vcf_file = pjoin(tmpdir,sample_name+'.cnv.vcf.gz')
        assert compare_zipped_files(out_vcf_file, expected_out_vcf)
        