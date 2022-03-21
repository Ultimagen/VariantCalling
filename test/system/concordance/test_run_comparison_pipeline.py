import os
import unittest
import sys
from os.path import dirname


import python.pipelines.pathmagic
sys.path.append(dirname(python.pipelines.pathmagic.__file__))

from python.pipelines import run_comparison_pipeline
from ugvc.concordance.concordance_utils import read_hdf
from test import test_dir, get_resource_dir, make_test_outputs_dir


class TestRunComparisonPipeline(unittest.TestCase):
    inputs_dir = get_resource_dir(__name__)
    test_outputs_dir = make_test_outputs_dir(__name__)
    general_inputs_dir = f'{test_dir}/resources/general/chr1_head'

    def test_run_comparison_pipeline(self):
        output_file = f'{self.test_outputs_dir}/HG00239.vcf.gz'
        os.makedirs(dirname(output_file), exist_ok=True)

        run_comparison_pipeline.main(
            ['--n_parts', '0',
             '--hpol_filter_length_dist', '12', '10',
             '--input_prefix', f'{self.inputs_dir}/004797-UGAv3-51.filtered.chr1_1_1000000',
             '--output_file', f'{self.test_outputs_dir}/004797-UGAv3-51.comp.h5',
             '--output_interval', f'{self.test_outputs_dir}/004797-UGAv3-51.comp.bed',
             '--gtr_vcf', f'{self.inputs_dir}/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.chr1_1_1000000.vcf.gz',
             '--highconf_intervals',
             f'{self.inputs_dir}/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark_noinconsistent.chr1_1_1000000.bed',
             '--runs_intervals', f'{self.general_inputs_dir}/hg38_runs.conservative.bed',
             '--reference', f'{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta',
             '--reference_dict', f'{self.general_inputs_dir}/Homo_sapiens_assembly38.dict',
             '--call_sample_name', 'UGAv3-51',
             '--truth_sample_name', 'HG004',
             '--ignore_filter_status',
             '--flow_order', 'TGCA',
             '--annotate_intervals', f'{self.general_inputs_dir}/LCR-hs38.bed',
             '--annotate_intervals', f'{self.general_inputs_dir}/exome.twist.bed',
             '--annotate_intervals', f'{self.general_inputs_dir}/mappability.0.bed',
             '--annotate_intervals', f'{self.general_inputs_dir}/hmers_7_and_higher.bed',
             '--n_jobs', '4',
             '--coverage_bw_all_quality', f'{self.inputs_dir}/004797-UGAv3-51.chr1.q0.Q0.l0.w1.depth.chr1_1_1000000.bw',
             '--coverage_bw_high_quality', f'{self.inputs_dir}/004797-UGAv3-51.chr1.q0.Q20.l0.w1.depth.chr1_1_1000000.bw'
             ])
        df = read_hdf(f'{self.test_outputs_dir}/004797-UGAv3-51.comp.h5', key='chr1')
        self.assertEqual({'tp': 346, 'fn': 29, 'fp': 27}, dict(df['classify'].value_counts()))
