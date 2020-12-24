from os.path import join as pjoin
import pathmagic
from pathmagic import PYTHON_TESTS_PATH
from os.path import dirname
from os.path import join as pjoin
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import python.modules.variant_annotation as annotation
import pandas as pd
from os.path import exists
import pysam
from collections import Counter
CLASS_PATH = "vcf_pipeline_utils"

def test_fix_errors():
    data = pd.read_hdf(pjoin(PYTHON_TESTS_PATH, CLASS_PATH, 'h5_file_unitest.h5'), key='concordance')
    #(TP, TP), (TP, None)
    df = vcf_pipeline_utils._fix_errors(data)
    assert all(df[((df['call'] == 'TP') & ((df['base'] == 'TP') | (df['base'].isna())))]['gt_ground_truth'].
               eq(df[(df['call'] == 'TP') & ((df['base'] == 'TP') | (df['base'].isna()))]['gt_ultima']))

    # (None, TP) (None,FN_CA)
    assert (df[(df['call'].isna()) & ((df['base'] == 'TP')
                                      | (df['base'] == 'FN_CA'))].size == 0)
    # (FP_CA,FN_CA), (FP_CA,None)
    temp_df = df.loc[(df['call'] == 'FP_CA') & ((df['base'] == 'FN_CA') | (
        df['base'].isna())), ['gt_ultima', 'gt_ground_truth']]
    assert all(temp_df.apply(lambda x: ((x['gt_ultima'][0] == x['gt_ground_truth'][0]) & (x['gt_ultima'][1] != x['gt_ground_truth'][1])) |
                             ((x['gt_ultima'][1] == x['gt_ground_truth'][1]) & (x['gt_ultima'][0] != x['gt_ground_truth'][0])) |
                             ((x['gt_ultima'][0] == x['gt_ground_truth'][1]) & (x['gt_ultima'][1] != x['gt_ground_truth'][0])) |
                             ((x['gt_ultima'][1] == x['gt_ground_truth'][0]) & (x['gt_ultima'][0] != x['gt_ground_truth'][1])), axis=1))

class TestVCFevalRun:
    ref_genome = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "sample.fasta")
    sample_calls = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "sample.sd.vcf.gz")
    truth_calls = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "gtr.sample.sd.vcf.gz")
    high_conf = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "highconf.interval_list")

    def test_vcfeval_run_ignore_filter(self, tmp_path):
        vcf_pipeline_utils.run_vcfeval_concordance(input_file=self.sample_calls,
                                                   truth_file=self.truth_calls,
                                                   output_prefix=str(tmp_path / "sample.ignore_filter"),
                                                   ref_genome=self.ref_genome,
                                                   comparison_intervals=self.high_conf,
                                                   input_sample="sm1",
                                                   truth_sample="HG001",   ignore_filter=True)
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(str(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")) as vcf:
            calls = Counter([x.info['CALL'] for x in vcf])
        assert calls == {'FP': 99, 'TP': 1}

    def test_vcfeval_run_use_filter(self, tmp_path):
        vcf_pipeline_utils.run_vcfeval_concordance(input_file=self.sample_calls,
                                                   truth_file=self.truth_calls,
                                                   output_prefix=str(tmp_path / "sample.use_filter"),
                                                   ref_genome=self.ref_genome,
                                                   comparison_intervals=self.high_conf,
                                                   input_sample="sm1",
                                                   truth_sample="HG001",   ignore_filter=False)
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(str(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")) as vcf:
            calls = Counter([x.info['CALL'] for x in vcf])
        assert calls == {'FP': 91, 'TP': 1, 'IGN': 8}

def test_annotate_concordance(mocker):
    spy = mocker.spy(annotation, 'classify_indel')
    data = pd.read_hdf('src/python_tests/h5_file_unitest.h5', key='concordance')
    vcf_pipeline_utils.annotate_concordance(data, '')
    spy.assert_called_once_with(data)