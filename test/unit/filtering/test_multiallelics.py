import pandas as pd
import pytest

import ugvc.filtering.multiallelics as tprep


def test_select_overlapping_variants():
    alleles = [["A", "T"], ["A", "T", "C"], ["A", "TA"], ["C", "A"], ["TA", "T"], ["T", "*", "A"]]
    positions = [10, 20, 30, 31, 40, 41]
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
    [("A", "G", "C"), (0, 1), False],
    [("A", "AG", "C"), (0, 1), True],
    [("A", "AG", "AC"), (1, 2), False],
    [("A", "C", "AC"), (0, 2), True],
]


@pytest.mark.parametrize("alleles,allele_indices,expected", testdata)
def test_is_indel_subset(alleles, allele_indices, expected):
    assert tprep.is_indel_subset(alleles, allele_indices) == expected


testdata = [
    [("A", "G", "C"), (0, 1), ((None,), (None,))],
    [("A", "AG", "C"), (0, 1), (("ins",), (1,))],
    [("A", "AG", "AC"), (1, 2), ((None,), (None,))],
    [("A", "C", "AC"), (0, 2), (("ins",), (1,))],
    [("A", "AC", "C"), (1, 2), (("del",), (1,))],
]


@pytest.mark.parametrize("alleles,allele_indices,expected", testdata)
def test_indel_classify_subset(alleles, allele_indices, expected):
    assert tprep.indel_classify_subset(alleles, allele_indices) == expected
