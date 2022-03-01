from os.path import join as pjoin
from pathmagic import PYTHON_TESTS_PATH
import pandas as pd
from python.pipelines import run_no_gt_report as run_no_gt_report
import mock
import subprocess


CLASS_PATH = "run_no_gt_report"

def test_allele_freq_hist():
    df = pd.DataFrame({'af':[0.1,0.2,0.3,0.4,0.5],
                       'variant_type':['snp','snp','h-indel','non-h-indel','non-h-indel']})
    result = run_no_gt_report.allele_freq_hist(df)
    assert len(result.keys()) == 3
    assert len(result['snp']) == 100
    assert len(result['h-indel']) == 100
    assert len(result['non-h-indel']) == 100
    assert result['snp'].iloc[10] == 1
    assert result['snp'].iloc[20] == 1
    assert sum(result['snp'] == 0) == 98


@mock.patch('subprocess.check_call')
def test_variant_eval_statistics(mocked_subprocess):
    output_prefix = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "collect_stats_unfiltered")
    data = run_no_gt_report.variant_eval_statistics('vcf', 'ref', 'db_snp', output_prefix,
                                                    ["exome","extended_exome","high_conf"]
                                                    )
    subprocess.check_call.assert_called_once_with(['gatk',
                                                   'VariantEval',
                                                   '--eval','vcf',
                                                   '--reference','ref',
                                                   '--dbsnp','db_snp',
                                                   '--output',
                                                   f'{output_prefix}.txt',
                                                   '--SELECT_NAMES', 'exome',
                                                   '--SELECT_NAMES', 'extended_exome',
                                                   '--SELECT_NAMES', 'high_conf',
                                                   '--SELECT_EXPS', 'vc.hasAttribute("exome")',
                                                   '--SELECT_EXPS', 'vc.hasAttribute("extended_exome")',
                                                   '--SELECT_EXPS', 'vc.hasAttribute("high_conf")'
                                                   ])
    for name in ['CompOverlap',
                 'CountVariants',
                 'IndelLengthHistogram',
                 'IndelSummary',
                 'MetricsCollection',
                 'TiTvVariantEvaluator',
                 'ValidationReport',
                 'VariantSummary']:
        assert data[name] is not None
        assert type(data[name]) == pd.core.frame.DataFrame

def test_insertion_deletion_statistics():
    df_path = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "df.hdf")
    df = pd.read_hdf(df_path, 'concordance')
    result = run_no_gt_report.insertion_deletion_statistics(df)
    assert result['homo'] is not None
    assert result['hete'] is not None

    assert result['homo'].shape == (4,12)
    assert result['hete'].shape == (4,12)

    assert result['homo'].loc['ins G',2] == 1
    result['homo']['ins G', 2] = 0
    assert all(result['hete'] == 0)
    assert all(result['hete'] == 0)


def test_snp_statistics():
    ref_fasta = pjoin(PYTHON_TESTS_PATH, "common", "sample.fasta")
    df_path = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "df.hdf")
    df = pd.read_hdf(df_path, 'chr20')
    result = run_no_gt_report.snp_statistics(df, ref_fasta)

    assert len(result) == 96
    assert result.loc[('CCA','A')] == 1
    assert result.loc[('CCA','T')] == 1
    assert result.loc[('AAT','G')] == 1
    assert result.loc[('ACT','T')] == 2
