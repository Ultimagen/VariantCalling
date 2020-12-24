import pytest
import mock
import pathmagic
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import python.modules.variant_annotation as annotation
import pandas as pd



def test_fix_errors():
    data = pd.read_hdf('src/python_tests/h5_file_unitest.h5', key='concordance')
    #(TP, TP), (TP, None)
    df = vcf_pipeline_utils._fix_errors(data)
    assert all(df[((df['call'] == 'TP') & ((df['base'] == 'TP') | (df['base'].isna())))]['gt_ground_truth'].
               eq(df[(df['call'] == 'TP') & ((df['base'] == 'TP') | (df['base'].isna()))][ 'gt_ultima']))

    # (None, TP) (None,FN_CA)
    assert (df[(df['call'].isna()) & ((df['base'] == 'TP') | (df['base'] == 'FN_CA'))].size == 0)
    # (FP_CA,FN_CA), (FP_CA,None)
    temp_df = df.loc[(df['call'] == 'FP_CA') & ((df['base'] == 'FN_CA') | (df['base'].isna())), ['gt_ultima','gt_ground_truth']]
    assert all(temp_df.apply(lambda x: ((x['gt_ultima'][0] == x['gt_ground_truth'][0]) & (x['gt_ultima'][1] != x['gt_ground_truth'][1])) |
            ((x['gt_ultima'][1] == x['gt_ground_truth'][1]) & (x['gt_ultima'][0] != x['gt_ground_truth'][0]))|
            ((x['gt_ultima'][0] == x['gt_ground_truth'][1]) & (x['gt_ultima'][1] != x['gt_ground_truth'][0]))|
            ((x['gt_ultima'][1] == x['gt_ground_truth'][0]) & (x['gt_ultima'][0] != x['gt_ground_truth'][1])), axis = 1))


def test_annotate_concordance(mocker):
    spy = mocker.spy(annotation, 'classify_indel')
    data = pd.read_hdf('src/python_tests/h5_file_unitest.h5', key='concordance')
    vcf_pipeline_utils.annotate_concordance(data, '')
    spy.assert_called_once_with(data)
