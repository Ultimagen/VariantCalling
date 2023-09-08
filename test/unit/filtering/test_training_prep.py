import pandas as pd

import ugvc.filtering.training_prep as tprep


def test_select_overlapping_variants():
    alleles = [["A", "T"], ["A", "T", "C"], ["A", "TA"], ["C", "A"], ["TA", "T"], ["T", "*", "A"]]
    positions = [10, 20, 30, 31, 40, 41]
    df = pd.DataFrame({"alleles": alleles, "pos": positions})
    result = tprep.select_overlapping_variants(df)
    assert result == [[1], [4, 5]]
