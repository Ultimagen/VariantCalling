import pickle
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np
import pysam

import ugvc.flow_format.flow_based_read as fbr
from ugvc.dna.format import DEFAULT_FLOW_ORDER

input_dir = get_resource_dir(__file__)


def test_matrix_from_qual_tp():
    data = [x for x in pysam.AlignmentFile(pjoin(input_dir, "chr9.sample.bam"))]
    expected = pickle.load(open(pjoin(input_dir, "matrices.trim.pkl"), "rb"))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12) for x in data
    ]
    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


def test_matrix_from_qual_tp_no_trim():
    data = [x for x in pysam.AlignmentFile(pjoin(input_dir, "chr9.sample.bam"))]
    expected = pickle.load(open(pjoin(input_dir, "matrices.pkl"), "rb"))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(
            x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12, spread_edge_probs=False
        )
        for x in data
    ]

    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.allclose(expected[i], rec._flow_matrix)


# test that spread probabilities on the first and last non-zero flow produces flat probabilities
# Since we read hmers 0-20 we expect P(0)=...P(20) = 1/21
def test_matrix_from_trimmed_read():
    data = [x for x in pysam.AlignmentFile(pjoin(input_dir, "trimmed_read.bam"))]
    flow_based_read = fbr.FlowBasedRead.from_sam_record(
        data[0], flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=20, spread_edge_probs=True
    )

    np.testing.assert_array_almost_equal(flow_based_read._flow_matrix[:, 2], np.ones(21) / 21, 0.0001)
    np.testing.assert_array_almost_equal(flow_based_read._flow_matrix[:, -1], np.ones(21) / 21, 0.0001)


def test_get_flow_matrix_column_for_base():
    data = [x for x in pysam.AlignmentFile(pjoin(input_dir, "chr9.sample.bam"))]
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(x, flow_order=DEFAULT_FLOW_ORDER, _fmt="cram", max_hmer_size=12) for x in data
    ]
    for rec in fbrs:
        for i in range(len(str(rec.record.query_sequence))):
            assert rec.get_flow_matrix_column_for_base(i)[0] == str(rec.record.query_sequence)[i]
            if i > 12 and i < len(str(rec.record.query_sequence)) - 12:
                assert (
                    rec.key[rec.base_to_flow_mapping[i]] == np.argmax(rec.get_flow_matrix_column_for_base(i)[1])
                ) or np.max(rec.get_flow_matrix_column_for_base(i)[1]) < 0.9
