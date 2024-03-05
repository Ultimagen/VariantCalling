from test import get_resource_dir

import numpy as np
import pandas as pd
import pytest

import ugvc.filtering.multiallelics as tprep


def test_select_overlapping_variants():
    alleles = [
        ["A", "T"],
        ["A", "T", "C"],
        ["A", "TA"],
        ["C", "A"],
        ["TA", "T"],
        ["T", "*", "A"],
        ["TAA", "T"],
        ["A", "AAAT"],
    ]
    positions = [10, 20, 30, 31, 40, 41, 60, 62]
    df = pd.DataFrame({"alleles": alleles, "pos": positions})
    result = tprep.select_overlapping_variants(df)
    assert result == [[1], [4, 5]]


testdata = [[(0, 1), (1, 2), (0, 0)], [(1, 1), (1, 2), (0, 0)], [(2, 2), (1, 2), (1, 1)], [(1, 2), (1, 2), (0, 1)]]


@pytest.mark.parametrize("orig_gt,allele_indices,expected", testdata)
def test_encode_gt_for_allele_subset(orig_gt, allele_indices, expected):
    assert tprep.encode_gt_for_allele_subset(orig_gt, allele_indices) == expected


def test_encode_gt_for_allele_subset_with_assertion():
    orig_gt = (0, 3)
    allele_indices = (1, 2)
    with pytest.raises(AssertionError):
        tprep.encode_gt_for_allele_subset(orig_gt, allele_indices)


testdata = [
    [(0, 10, 20), (0, 1), (0, 10, 20)],
    [(0, 10, 20, 40, 50, 60), (1, 2), (0, 30, 40)],
    [(0, 10, 20, 40, 50, 60, 100, 120, 130, 140), (0, 3), (0, 100, 140)],
]


@pytest.mark.parametrize("orig_pl,allele_indices,expected", testdata)
def test_select_pl_for_allele_subset(orig_pl, allele_indices, expected):
    assert tprep.select_pl_for_allele_subset(orig_pl, allele_indices) == expected


testdata = [
    [("A", "*", "C"), (0, 1), pd.Series([0]), True],
    [("A", "*", "C"), (0, 2), pd.Series([0]), False],
    [("A", "AG", "*"), (1, 2), pd.Series([0]), True],
]


@pytest.mark.parametrize("alleles,allele_indices,spandel,expected", testdata)
def test_is_indel_subset_with_spandel(alleles, allele_indices, spandel, expected):
    assert tprep.is_indel_subset(alleles, allele_indices, spandel=spandel) == expected


testdata = [
    [("A", "G", "C"), (0, 1), False],
    [("A", "AG", "C"), (0, 1), True],
    [("A", "AG", "AC"), (1, 2), False],
    [("A", "C", "AC"), (0, 2), True],
]


@pytest.mark.parametrize("alleles,allele_indices,expected", testdata)
def test_is_indel_subset(alleles, allele_indices, expected):
    assert tprep.is_indel_subset(alleles, allele_indices) == expected


testdata = [
    [("A", "G", "C"), (0, 1), (("NA",), (None,))],
    [("A", "AG", "C"), (0, 1), (("ins",), (1,))],
    [("A", "AG", "AC"), (1, 2), (("NA",), (None,))],
    [("A", "C", "AC"), (0, 2), (("ins",), (1,))],
    [("A", "AC", "C"), (1, 2), (("del",), (1,))],
]


@pytest.mark.parametrize("alleles,allele_indices,expected", testdata)
def test_indel_classify_subset(alleles, allele_indices, expected):
    assert tprep.indel_classify_subset(alleles, allele_indices) == expected


testdata = [
    [("A", "G", "*"), (0, 1), pd.Series({"x_il": (4, 5)}), (("NA",), (None,))],
    [("A", "AG", "*"), (0, 1), pd.Series({"x_il": (4, 5)}), (("ins",), (1,))],
    [("A", "C", "*"), (0, 2), pd.Series({"x_il": (4, 5)}), (("del",), (4,))],
    [("A", "AC", "*"), (1, 2), pd.Series({"x_il": (4, 5)}), (("del",), (4,))],
]


@pytest.mark.parametrize("alleles,allele_indices,spandel, expected", testdata)
def test_indel_classify_subset_with_spandel(alleles, allele_indices, spandel, expected):
    assert tprep.indel_classify_subset(alleles, allele_indices, spandel=spandel) == expected


ref = "A" * 20 + "G" + "ACCGCT" + "A" * 20
spandel = pd.Series({"alleles": ("GA", "G"), "pos": 21})
testdata = [
    [("A", "C"), (0, 1), ref, 22, spandel, (".", 0)],
    [("A", "C", "CA"), (1, 2), ref, 22, spandel, (".", 0)],
    [("A", "C", "CC"), (1, 2), ref, 22, spandel, ("C", 4)],
    [("A", "CC", "C"), (1, 2), ref, 22, spandel, ("C", 4)],
    [("A", "C", "*"), (1, 2), ref, 22, spandel, ("C", 3)],
]


@pytest.mark.parametrize("alleles,allele_indices,ref,pos,spandel,expected", testdata)
def test_classify_hmer_indel_relative(alleles, allele_indices, ref, pos, spandel, expected):
    assert tprep.classify_hmer_indel_relative(alleles, allele_indices, ref, pos, spandel=spandel) == expected


testdata = [
    [(-1, -1), (0, 1), (-1, -1)],
    [(-1, 2), (0, 1), (-1, 2)],
    [(1, 1), (0, 1), (1, 1)],
    [(1, 2), (0, 1), (1, 1)],
    [(1, 2), (1, 2), (0, 1)],
]


@pytest.mark.parametrize("label,allele_indices,expected", testdata)
def test_encode_label(label, allele_indices, expected):
    assert tprep.encode_label(label, allele_indices) == expected


def test_cleanup_multiallelics():
    inputs_dir = get_resource_dir(__file__)

    input = pd.DataFrame(pd.read_hdf(f"{inputs_dir}/cleanup_multiallelics_input.h5"))
    expected = pd.DataFrame(pd.read_hdf(f"{inputs_dir}/cleanup_multiallelics_expected.h5"))
    result = tprep.cleanup_multiallelics(input)
    pd.testing.assert_frame_equal(result, expected)
    result = result.loc[result["label"].apply(lambda x: x in {(0, 1), (1, 1), (0, 0)})]
    assert np.all(result.loc[(result["x_hil"] != (None,)) & (result["x_hil"] != (0,)), "variant_type"] == "h-indel")
    assert np.all(result.loc[result["x_hil"] == (None,), "variant_type"] != "h-indel")
    select = ~pd.isnull(result["str"])
    result = result.loc[select]
    assert np.all(
        (
            (result["rpa"].apply(lambda x: x[0]) > result["rpa"].apply(lambda x: x[1]))
            & (result["x_ic"].apply(lambda x: x[0]) == "del")
        )
        | (
            (result["rpa"].apply(lambda x: x[0]) < result["rpa"].apply(lambda x: x[1]))
            & (result["x_ic"].apply(lambda x: x[0]) == "ins")
        )
    )
