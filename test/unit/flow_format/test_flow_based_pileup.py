from unittest.mock import MagicMock

import pysam
import pytest

from ugvc.flow_format.flow_based_pileup import FlowBasedPileupColumn
from ugvc.flow_format.flow_based_read import FlowBasedRead


@pytest.fixture
def pileup_column():
    pileup_column = MagicMock(spec=pysam.PileupColumn)
    pileup_column.pileups = [
        MagicMock(query_position_or_next=0, alignment=MagicMock(query_name="read1")),
        MagicMock(query_position_or_next=1, alignment=MagicMock(query_name="read2")),
    ]
    return pileup_column


@pytest.fixture
def flow_reads_dict():
    flow_read1 = MagicMock(spec=FlowBasedRead)
    flow_read1.get_flow_matrix_column_for_base.return_value = (1, [0.1, 0.9])
    flow_read2 = MagicMock(spec=FlowBasedRead)
    flow_read2.get_flow_matrix_column_for_base.return_value = (2, [0.2, 0.8])
    return {"read1": flow_read1, "read2": flow_read2}


def test_flow_based_pileup_column_initialization(pileup_column, flow_reads_dict):
    fbp_column = FlowBasedPileupColumn(pileup_column, flow_reads_dict)
    assert fbp_column.pc == pileup_column
    assert fbp_column.flow_reads_dict == flow_reads_dict


def test_fetch_hmer_qualities(pileup_column, flow_reads_dict):
    fbp_column = FlowBasedPileupColumn(pileup_column, flow_reads_dict)
    result = fbp_column.fetch_hmer_qualities()
    expected = [(1, [0.1, 0.9]), (2, [0.2, 0.8])]
    assert result == expected
