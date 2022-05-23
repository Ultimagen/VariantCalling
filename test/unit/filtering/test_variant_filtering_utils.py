import pickle
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np
import pandas as pd
import pytest

import ugvc.filtering.blacklist
from ugvc.filtering import variant_filtering_utils

inputs_dir = get_resource_dir(__file__)


def test_add_testing_train_split_column():
    concordance_df = pd.DataFrame({'qual':np.arange(100)})
    concordance_df['group'] = 'snp'

    concordance_df = variant_filtering_utils.add_testing_train_split_column(concordance_df,
        training_groups_column='group',
        test_train_split_column = 'tts',
        gtr_column = 'group',
        min_test_set = 20, 
        max_train_set = 50, 
        test_set_fraction = 0.5)
    assert concordance_df['tts'].sum() == 50

    concordance_df = variant_filtering_utils.add_testing_train_split_column(concordance_df,
        training_groups_column='group',
        test_train_split_column = 'tts',
        gtr_column = 'group',
        min_test_set = 20, 
        max_train_set = 50, 
        test_set_fraction = 0.8)
    assert concordance_df['tts'].sum() == 20
    concordance_df = variant_filtering_utils.add_testing_train_split_column(concordance_df,
        training_groups_column='group',
        test_train_split_column = 'tts',
        gtr_column = 'group',
        min_test_set = 90, 
        max_train_set = 90, 
        test_set_fraction = 0.5)
    assert concordance_df['tts'].sum() == 10

# tests determinism of test train split
def test_add_testing_train_split_column_deterministic():
    concordance_df = pd.DataFrame({'qual':np.arange(100)})
    concordance_df['group'] = 'snp'
    concordance_df = variant_filtering_utils.add_testing_train_split_column(concordance_df,
        training_groups_column='group',
        test_train_split_column = 'tts',
        gtr_column = 'group',
        min_test_set = 20, 
        max_train_set = 50)
    assert concordance_df['tts'].sum() == 50
    assert concordance_df['tts'].to_numpy().nonzero()[0].min() == 0
    assert concordance_df['tts'].to_numpy().nonzero()[0].max() == 96    
    assert concordance_df['tts'].to_numpy().nonzero()[0].mean() == 45.3   


def test_blacklist_cg_insertions():
    rows = pd.DataFrame(
        {
            "alleles": [("C", "T"), ("CCG", "C"), ("G", "GGC",)],
            "filter": ["PASS", "PASS", "PASS"],
        }
    )
    rows = ugvc.filtering.blacklist.blacklist_cg_insertions(rows)
    filters = list(rows)
    assert filters == ["PASS", "CG_NON_HMER_INDEL", "CG_NON_HMER_INDEL"]


def test_merge_blacklist():
    list1 = pd.Series(["PASS", "FAIL", "FAIL"])
    list2 = pd.Series(["PASS", "FAIL1", "PASS"])
    merge_list = list(ugvc.filtering.blacklist.merge_blacklists([list1, list2]))
    assert merge_list == ["PASS;PASS", "FAIL;FAIL1", "FAIL;PASS"]


def test_read_blacklist():
    blacklist = pickle.load(open(pjoin(inputs_dir, "blacklist.test.pkl"), "rb"))
    descriptions = [str(x) for x in blacklist]
    assert descriptions == [
        "ILLUMINA_FP: GTR error with 1000 elements",
        "COHORT_FP: HMER indel FP seen in cohort with 1000 elements",
    ]


def test_apply_blacklist():
    df = pd.read_hdf(pjoin(inputs_dir, "test.df.h5"), key="variants")
    blacklist = pickle.load(open(pjoin(inputs_dir, "blacklist.test.pkl"), "rb"))
    blacklists_applied = [x.apply(df) for x in blacklist]
    vcs = [x.value_counts() for x in blacklists_applied]
    assert (
        vcs[0]["PASS"] == 6597
        and vcs[0]["ILLUMINA_FP"] == 6
        and vcs[1]["PASS"] == 6477
        and vcs[1]["COHORT_FP"] == 126
    )


def test_fpr_tree_score_mapping():
    tree_scores = np.arange(0.01, 0.1, 0.01)
    labels = np.array(["tp", "fp", "fn", "fp", "tp", "fp", "tp", "tp", "fp"])
    test_train_split = np.array(
        [True, False, True, False, True, True, False, False, False, True]
    )
    interval_size = 10 ** 6

    sorted_ts, fpr = variant_filtering_utils.fpr_tree_score_mapping(
        tree_scores, labels, test_train_split, interval_size
    )
    assert all(np.around(fpr, 1) == [8, 8, 6, 6, 4, 4, 2, 2, 2])
    assert all(np.around(sorted_ts, 2) == np.around(np.arange(0.01, 0.1, 0.01), 2))


def test_tree_score_to_fpr():
    df = pd.DataFrame({"group": np.repeat(np.array([["h-mer", "snp"]]), 5)})
    prediction_score = pd.Series(np.arange(1, 0, -0.1))
    tree_score_fpr = {
        "snp": pd.DataFrame(
            {"tree_score": np.arange(0, 1, 0.1), "fpr": np.arange(0, 10, 1)}
        ),
        "h-mer": pd.DataFrame(
            {"tree_score": np.arange(0, 1, 0.2), "fpr": np.arange(0, 10, 2)}
        ),
    }
    fpr_values = variant_filtering_utils.tree_score_to_fpr(
        df, prediction_score, tree_score_fpr
    )
    print(fpr_values)
    assert all(
        np.around(fpr_values, 1) == [8.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    )


def test_validate_data():
    test1 = np.array([[0, 1], [1, 2]])
    variant_filtering_utils._validate_data(test1)
    test2 = np.array([[0, 1], [1, np.NaN]])
    with pytest.raises(AssertionError):
        variant_filtering_utils._validate_data(test2)
    test1 = pd.DataFrame([[0, 1], [1, 2]])
    variant_filtering_utils._validate_data(test1)
    test2 = pd.DataFrame([[0, 1], [1, np.NaN]])
    with pytest.raises(AssertionError):
        variant_filtering_utils._validate_data(test2)
    test1 = pd.Series([0, 1])
    variant_filtering_utils._validate_data(test1)
    test2 = pd.Series([1, np.NaN])
    with pytest.raises(AssertionError):
        variant_filtering_utils._validate_data(test2)
