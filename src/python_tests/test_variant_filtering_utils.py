import pytest
import pathmagic
import pandas as pd
import pickle
from os.path import join as pjoin
from pathmagic import PYTHON_TESTS_PATH
from python.pipelines import variant_filtering_utils


def test_blacklist_cg_insertions():
    rows = pd.DataFrame(
        {'alleles': [('C', 'T'),
                     ('CCG', 'C'),
                     ('G', 'GGC',)], 'filter': ['PASS', 'PASS', 'PASS']})
    rows = variant_filtering_utils.blacklist_cg_insertions(rows)
    filters = list(rows)
    assert filters == ['PASS', 'CG_NON_HMER_INDEL', 'CG_NON_HMER_INDEL']


def test_merge_blacklist():
    list1 = pd.Series(["PASS", "FAIL", "FAIL"])
    list2 = pd.Series(["PASS", "FAIL1", "PASS"])
    merge_list = list(variant_filtering_utils.merge_blacklists([list1, list2]))
    assert merge_list == ["PASS;PASS", "FAIL;FAIL1", "FAIL;PASS"]


def test_read_blacklist():
    blacklist = pickle.load(
        open(pjoin(PYTHON_TESTS_PATH, "blacklist.test.pkl"), "rb"))
    descriptions = [str(x) for x in blacklist]
    assert descriptions == ['ILLUMINA_FP: GTR error with 1000 elements',
                            'COHORT_FP: HMER indel FP seen in cohort with 1000 elements']


def test_apply_blacklist():
    df = pd.read_hdf(pjoin(PYTHON_TESTS_PATH, "test.df.h5"), key="variants")
    blacklist = pickle.load(
        open(pjoin(PYTHON_TESTS_PATH, "blacklist.test.pkl"), "rb"))
    blacklists_applied = [x.apply(df) for x in blacklist]
    vcs = [x.value_counts() for x in blacklists_applied]
    assert vcs[0]['PASS'] == 6597 and vcs[0]['ILLUMINA_FP'] == 6 \
        and vcs[1]['PASS'] == 6477 and vcs[1]['COHORT_FP'] == 126

