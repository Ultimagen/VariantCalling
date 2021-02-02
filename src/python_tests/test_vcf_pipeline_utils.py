import pathmagic
from pathmagic import PYTHON_TESTS_PATH, COMMON
from os.path import join as pjoin
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import python.modules.variant_annotation as annotation
import subprocess
import pandas as pd
from os.path import exists
import pysam
import os
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
    ref_genome = pjoin(PYTHON_TESTS_PATH, COMMON, "sample.fasta")
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


def test_intersect_bed_files(mocker, tmp_path):
    bed1 = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed1.bed")
    bed2 = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed2.bed")
    output_path = pjoin(tmp_path, "output.bed")

    spy_subprocess = mocker.spy(subprocess, 'call')
    vcf_pipeline_utils.intersect_bed_files(bed1, bed2, output_path)
    spy_subprocess.assert_called_once_with(['bedtools', 'intersect', '-a', bed1,'-b',bed2], stdout = mocker.ANY)

    assert exists(output_path)


# def test_get_interval_length(mocker, tmp_path):
#     bed1 = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed1.bed")
#     bed2 = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed2.bed")
#
#     spy_intersect = mocker.spy(vcf_pipeline_utils, 'intersect_bed_files')
#     spy_length = mocker.spy(vcf_pipeline_utils, 'bed_file_length')
#
#     intersect_length = vcf_pipeline_utils.get_interval_length(bed1, bed2)
#     assert intersect_length == 2705
#
#     spy_intersect.assert_called_once_with(bed1, bed2, mocker.ANY)
#     temp_file = spy_intersect.call_args[0][2]
#     spy_length.assert_called_once_with(temp_file)

def test_bed_file_length():
    bed1 = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed1.bed")
    result = vcf_pipeline_utils.bed_file_length(bed1)
    assert result == 3026

def test_IntervalFile_init_bed_input(mocker):
    bed1 = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed1.bed")
    ref_genome = pjoin(PYTHON_TESTS_PATH, COMMON, "sample.fasta")
    interval_list_path = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "bed1.interval_list")
    spy_subprocess = mocker.spy(subprocess, 'check_call')

    intervalFile = vcf_pipeline_utils.IntervalFile(bed1, ref_genome)

    assert intervalFile.as_bed_file() == bed1
    assert intervalFile.as_interval_list_file() == interval_list_path
    assert exists(interval_list_path)
    assert not intervalFile.is_none()
    spy_subprocess.assert_called_once_with(['picard', 'BedToIntervalList',f'I={bed1}', f'O={interval_list_path}', f'SD={ref_genome}.dict'])
    os.remove(interval_list_path)

def test_IntervalFile_init_interval_list_input(mocker):
    interval_list = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "interval_list1.interval_list")
    ref_genome = pjoin(PYTHON_TESTS_PATH, COMMON, "sample.fasta")
    bed_path = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "interval_list1.bed")
    spy_subprocess = mocker.spy(subprocess, 'check_call')

    intervalFile = vcf_pipeline_utils.IntervalFile(interval_list, ref_genome)

    assert intervalFile.as_bed_file() == bed_path
    assert intervalFile.as_interval_list_file() == interval_list
    assert exists(bed_path)
    assert not intervalFile.is_none()
    spy_subprocess.assert_called_once_with(['picard', 'IntervalListToBed',f'I={interval_list}', f'O={bed_path}'])
    os.remove(bed_path)


def test_IntervalFile_init_error():
    ref_genome = pjoin(PYTHON_TESTS_PATH, COMMON, "sample.fasta")
    intervalFile = vcf_pipeline_utils.IntervalFile(ref_genome, ref_genome)
    assert intervalFile.as_bed_file() is None
    assert intervalFile.as_interval_list_file() is None
    assert intervalFile.is_none()

