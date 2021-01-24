import pathmagic
import pandas as pd
import numpy as np
import pickle
from os.path import join as pjoin
from pathmagic import PYTHON_TESTS_PATH
from python.pipelines import variant_filtering_utils
CLASS_PATH="variant_filtering"

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
        open(pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "blacklist.test.pkl"), "rb"))
    descriptions = [str(x) for x in blacklist]
    assert descriptions == ['ILLUMINA_FP: GTR error with 1000 elements',
                            'COHORT_FP: HMER indel FP seen in cohort with 1000 elements']


def test_apply_blacklist():
    df = pd.read_hdf(pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "test.df.h5"), key="variants")
    blacklist = pickle.load(
        open(pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "blacklist.test.pkl"), "rb"))
    blacklists_applied = [x.apply(df) for x in blacklist]
    vcs = [x.value_counts() for x in blacklists_applied]
    assert vcs[0]['PASS'] == 6597 and vcs[0]['ILLUMINA_FP'] == 6 \
        and vcs[1]['PASS'] == 6477 and vcs[1]['COHORT_FP'] == 126


def test_fpr_calc():
    tree_scores = np.arange(0.01,0.1,0.01)
    labels = np.array(['tp','fp','fn','fp','tp','fp','tp','tp','fp'])
    test_train_split = np.array([True,False,True,False,True,True,False,False,False,True])
    interval_size = 10**6

    fpr = variant_filtering_utils.fpr_calc(tree_scores,labels,test_train_split,interval_size)
    assert all(fpr==(pd.Series([0,2, 2, 4, 4, 6,6,6, 8])))

def test_score_to_fpr():
    df = pd.DataFrame({'group': np.repeat('snp', 10)})
    prediction_score = pd.Series(np.arange(1,0,-0.1))
    tree_score_fpr = {'snp':pd.DataFrame({'tree_score':np.arange(0,1,0.1),'fpr':np.arange(0,10,1)})}
    fpr_values = variant_filtering_utils.score_to_fpr(df, prediction_score, tree_score_fpr)
    assert all(fpr_values == [9,9,8,7,6,5,4,3,2,1])
