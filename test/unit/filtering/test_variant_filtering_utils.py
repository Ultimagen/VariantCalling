import pickle
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np
import pandas as pd
import pytest

import ugvc.filtering.blacklist
from ugvc.filtering import variant_filtering_utils

inputs_dir = get_resource_dir(__file__)


def test_blacklist_cg_insertions():
    rows = pd.DataFrame(
        {
            "alleles": [
                ("C", "T"),
                ("CCG", "C"),
                (
                    "G",
                    "GGC",
                ),
            ],
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
        vcs[0]["PASS"] == 6597 and vcs[0]["ILLUMINA_FP"] == 6 and vcs[1]["PASS"] == 6477 and vcs[1]["COHORT_FP"] == 126
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


def test_get_empty_recall_precision():
    result = variant_filtering_utils.get_empty_recall_precision()
    expected_result = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "initial_tp": 0,
        "initial_fp": 0,
        "initial_fn": 0,
        "initial_precision": 1.0,
        "initial_recall": 1.0,
        "initial_f1": 1.0,
    }
    assert result == expected_result


def test_get_empty_recall_precision_curve():
    result = variant_filtering_utils.get_empty_recall_precision_curve()
    expected_result = {
        "threshold": 0,
        "predictions": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    assert result == expected_result


def test_get_concordance_metrics():
    predictions = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.2, 0.8, 0.4, 0.6, 0.3])
    truth = np.array([0, 1, 1, 0, 1])
    fn_mask = np.array([False, False, True, False, False])
    return_metrics = True
    return_curves = False

    result = variant_filtering_utils.get_concordance_metrics(
        predictions, scores, truth, fn_mask, return_metrics, return_curves
    )

    expected_metrics_df = pd.DataFrame(
        {
            "tp": [1],
            "fp": [1],
            "fn": [2],
            "precision": [0.5],
            "recall": [0.3333333],
            "f1": [0.4],
            "initial_tp": [2],
            "initial_fp": [2],
            "initial_fn": [1],
            "initial_precision": [0.5],
            "initial_recall": [0.6666666666666666],
            "initial_f1": [0.5714285714285714],
        }
    )

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected_metrics_df)


def test_get_concordance_metrics_no_return():
    predictions = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.2, 0.8, 0.4, 0.6, 0.3])
    truth = np.array([0, 1, 1, 0, 1])
    fn_mask = np.array([False, False, True, False, False])
    return_metrics = False
    return_curves = False

    with pytest.raises(AssertionError):
        variant_filtering_utils.get_concordance_metrics(
            predictions, scores, truth, fn_mask, return_metrics, return_curves
        )
