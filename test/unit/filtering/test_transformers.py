import pandas as pd
import pytest

from ugvc.filtering.transformers import (
    allele_encode,
    encode_labels,
    gt_encode,
    ins_del_encode,
    motif_encode_left,
    motif_encode_right,
    tuple_break,
    tuple_break_second,
    tuple_break_third,
)


class TestTransformers:
    def setup(self):
        self.sample_tuple = (1, 2, 3)
        self.sample_motif = "ATGC"
        self.sample_base = "A"
        self.sample_vtype = "single_sample"
        self.sample_df = pd.DataFrame(
            {
                "sor": [1, 2, 3],
                "dp": [4, 5, 6],
                "alleles": ["A", "T", "G"],
                "x_hin": [(1, "A"), (2, "T"), (3, "G")],
                "x_hil": [(4, 5), (6, 7), (8, 9)],
                "x_il": [(10, 11), (12, 13), (14, 15)],
                "indel": ["ins", "del", "NA"],
                "x_ic": [(16, 17), (18, 19), (20, 21)],
                "x_lm": ["ATG", "CGT", "GCA"],
                "x_rm": ["TGC", "GTA", "ATC"],
                "x_css": [(22, 23), (24, 25), (26, 27)],
                "x_gcc": [28, 29, 30],
            }
        )

    def test_tuple_break(self):
        result = tuple_break(self.sample_tuple)
        assert result == 1

    def test_tuple_break_second(self):
        result = tuple_break_second(self.sample_tuple)
        assert result == 2

    def test_tuple_break_third(self):
        result = tuple_break_third(self.sample_tuple)
        assert result == 3

    def test_motif_encode_left(self):
        result = motif_encode_left(self.sample_motif)
        assert result == 4321

    def test_motif_encode_right(self):
        result = motif_encode_right(self.sample_motif)
        assert result == 1234

    def test_allele_encode(self):
        result = allele_encode(self.sample_base)
        assert result == 1

    def test_gt_encode(self):
        result = gt_encode((1, 1))
        assert result == 1
        result = gt_encode((1, 0))
        assert result == 0

    def test_ins_del_encode(self):
        result = ins_del_encode("ins")
        assert result == -1
        result = ins_del_encode("del")
        assert result == 1
        result = ins_del_encode("NA")
        assert result == 0

    def test_encode_labels(self):
        labels = [(0, 1), (0, 0), (1, 0), (1, 1)]
        encoded_labels = encode_labels(labels)
        expected_result = [1, 0, 1, 2]
        assert encoded_labels == expected_result
        with pytest.raises(ValueError):
            encode_labels([(0, 1, 2), (0, 0, 1)])  # type: ignore
        with pytest.raises(ValueError):
            encode_labels([(0, 2), (1, 2)])
