from os.path import join as pjoin
import pathmagic
from pathmagic import PYTHON_TESTS_PATH
import python.modules.flowBasedRead as fbr
import pysam
import numpy as np
import pickle

from os.path import dirname
CLASS_PATH = "flow_based_read"

def test_matrix_from_qual_tp():
    data = [x for x in pysam.AlignmentFile(
        pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "chr9.sample.bam"))]
    expected = pickle.load(
        open(pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "matrices.pkl"), "rb"))
    fbrs = [fbr.FlowBasedRead.from_sam_record(
        x, flow_order='TGCA', format='cram', max_hmer_size=12) for x in data]

    for i, rec in enumerate(fbrs):
        assert rec.key.sum() == len(rec.record.query_sequence)
        if i < len(expected):
            assert(np.all(expected[i] == rec._flow_matrix))
