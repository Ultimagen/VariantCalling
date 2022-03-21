import os
import unittest
from os.path import dirname
import sys

import python.pipelines.pathmagic
sys.path.append(dirname(python.pipelines.pathmagic.__file__))

import python.vcftools as vcftools
from python.pipelines import filter_variants_pipeline
from test import test_dir, get_resource_dir, make_test_outputs_dir


class TestFilterVariantPipeline(unittest.TestCase):
    inputs_dir = get_resource_dir(__file__)
    test_outputs_dir = make_test_outputs_dir(__file__)
    general_inputs_dir = f'{test_dir}/resources/general/chr1_head'

    def test_filter_variants_pipeline(self):
        output_file = f'{self.test_outputs_dir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.vcf.gz'
        filter_variants_pipeline.main(
            ['--input_file', f'{self.inputs_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz',
             '--model_file', f'{self.inputs_dir}/004777-X0024.model_rf_model_ignore_gt_incl_hpol_runs.pkl',
             '--model_name', 'rf_model_ignore_gt_incl_hpol_runs',
             '--runs_file', f'{self.general_inputs_dir}/hg38_runs.conservative.bed',
             '--hpol_filter_length_dist', '12', '10',
             '--reference_file', f'{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta',
             '--blacklist_cg_insertions',
             '--annotate_intervals', f'{self.general_inputs_dir}/LCR-hs38.bed',
             '--annotate_intervals', f'{self.general_inputs_dir}/exome.twist.bed',
             '--annotate_intervals', f'{self.general_inputs_dir}/mappability.0.bed',
             '--annotate_intervals', f'{self.general_inputs_dir}/hmers_7_and_higher.bed',
             '--is_mutect',
             '--output_file', output_file,
             '--blacklist', f'{self.inputs_dir}/blacklist_example.chr1_1_1000000.pkl'
             ])

        df = vcftools.get_vcf_df(output_file)
        self.assertEqual({'LOW_SCORE': 51, 'PASS': 857}, dict(df['filter'].value_counts()))
