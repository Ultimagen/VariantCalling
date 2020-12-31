import pytest
import pathmagic
from python.pipelines import vc_pipeline_utils
from pathmagic import PYTHON_TESTS_PATH
from os.path import join as pjoin
picard_file = pjoin(PYTHON_TESTS_PATH, "140479-BC21.alignment_summary_metrics")


def test_parse_cvg_metrics():
    metric_df = vc_pipeline_utils.parse_cvg_metrics(picard_file)[1]
    histogram_df = vc_pipeline_utils.parse_cvg_metrics(picard_file)[2]
    assert vc_pipeline_utils.parse_cvg_metrics(picard_file)[0] == 'AlignmentSummaryMetrics'
    assert metric_df.TOTAL_READS[0] == 640936116
    assert histogram_df.READ_LENGTH[4] == 29
