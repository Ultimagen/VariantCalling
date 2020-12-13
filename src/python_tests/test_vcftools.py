from os.path import join as pjoin
import pathmagic
from pathmagic import PYTHON_TESTS_PATH
import python.vcftools as vcftools
import pandas as pd
from os.path import dirname
from os.path import join as pjoin

def test_bed_files_output():
    # snp_fp testing
    data = pd.read_hdf(pjoin(PYTHON_TESTS_PATH, 'BC10.chr1.h5'), key='concordance')
    snp_fp = vcftools.FilterWrapper(data).get_SNP().get_fp().get_df()
    assert all([x == False for x in snp_fp['indel']])
    assert all([x == 'fp' for x in snp_fp['classify']])

    # snp_fn testing
    snp_fn = vcftools.FilterWrapper(data).get_SNP().get_fn().get_df()
    assert all([x == False for x in snp_fn['indel']])
    assert all([row['classify'] == 'fn' or (row['classify'] == 'tp' and
                                            (row['filter'] == 'LOW_SCORE' or
                                             row['filter'] == 'HPOL_RUN;LOW_SCORE') and
                                            (row['filter'] != 'PASS' and row['filter'] != 'HPOL_RUN'))
                for index, row in snp_fn.iterrows()])

    # hmer
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer().get_df()
    assert all(hmer_fn['indel'])
    assert all([x > 0 for x in hmer_fn['hmer_indel_length']])
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer(
        val_start=1, val_end=1).get_df()
    assert all(hmer_fn['indel'])
    assert all([x == 1 for x in hmer_fn['hmer_indel_length']])
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer(
        val_start=3, val_end=10).get_df()
    assert all(hmer_fn['indel'])
    assert all([x >= 3 & x <= 10 for x in hmer_fn['hmer_indel_length']])

    # hmer_fp testing
    hmer_fp = vcftools.FilterWrapper(data).get_h_mer().get_fp().get_df()
    assert all(hmer_fp['indel'])
    assert all([x > 0 for x in hmer_fp['hmer_indel_length']])
    assert all([x == 'fp' for x in hmer_fp['classify']])

    # hmer_fn testing
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer().get_fn().get_df()
    assert all(hmer_fn['indel'])
    assert all([x > 0 for x in hmer_fn['hmer_indel_length']])
    assert all([row['classify'] == 'fn' or (row['classify'] == 'tp' and
                                            (row['filter'] == 'LOW_SCORE' or
                                             row['filter'] == 'HPOL_RUN;LOW_SCORE') and
                                            (row['filter'] != 'PASS' and row['filter'] != 'HPOL_RUN'))
                for index, row in hmer_fn.iterrows()])

    # non_hmer_fp testing
    non_hmer_fp = vcftools.FilterWrapper(
        data).get_non_h_mer().get_fp().get_df()
    assert all(non_hmer_fp['indel'])
    assert all([x == 0 for x in non_hmer_fp['hmer_indel_length']])
    assert all([x == 'fp' for x in non_hmer_fp['classify']])

    # non_hmer_fn testing
    non_hmer_fn = vcftools.FilterWrapper(
        data).get_non_h_mer().get_fn().get_df()
    assert all(non_hmer_fn['indel'])
    assert all([x == 0 for x in non_hmer_fn['hmer_indel_length']])
    assert all([row['classify'] == 'fn' or (row['classify'] == 'tp' and
                                            (row['filter'] == 'LOW_SCORE' or
                                             row['filter'] == 'HPOL_RUN;LOW_SCORE') and
                                            (row['filter'] != 'PASS' and row['filter'] != 'HPOL_RUN'))
                for index, row in non_hmer_fn.iterrows()])


def test_bed_output_when_no_tree_score():# testing the case when there is no tree_score and there is blacklist
    data = pd.read_hdf(pjoin(dirname(__file__), 'exome.h5'), key='concordance')
    df = vcftools.FilterWrapper(data)
    result = dict(df.get_fn().BED_format(kind="fn").get_df()['itemRgb'].value_counts())
    expected_result = {vcftools.FilteringColors.BLACKLIST: 169, 
                        vcftools.FilteringColors.CLEAR: 89, vcftools.FilteringColors.BORDERLINE: 39}
    for k in result:
        assert result[k] == expected_result[k]
    df = vcftools.FilterWrapper(data)
    # since there is no tree_score all false positives should be the same color
    result = dict(df.get_fp().BED_format(kind="fp").get_df()['itemRgb'].value_counts())
    assert len(result.keys()) == 1
