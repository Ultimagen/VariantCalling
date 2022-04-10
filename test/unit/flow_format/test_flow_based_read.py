import pickle
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np
import pysam

import ugvc.flow_format.flowBasedRead as fbr
from ugvc.dna.format import DEFAULT_FLOW_ORDER

input_dir = get_resource_dir(__file__)


def test_matrix_from_qual_tp():
    data = [x for x in pysam.AlignmentFile(pjoin(input_dir, "chr9.sample.bam"))]
    expected = pickle.load(open(pjoin(input_dir, "matrices.pkl"), "rb"))
    fbrs = [
        fbr.FlowBasedRead.from_sam_record(
            x, flow_order=DEFAULT_FLOW_ORDER, format="cram", max_hmer_size=12
        )
        for x in data
    ]

    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert np.all(expected[i] == rec._flow_matrix)
