from test import get_resource_dir, test_dir
from ugvc.utils import picard_metrics_utils

inputs_dir = get_resource_dir(__file__)

def test_parse_cvg_metrics():
    picard_file = str(inputs_dir / "140479-BC21.alignment_summary_metrics")
    metric_df = picard_metrics_utils.parse_cvg_metrics(picard_file)[1]
    histogram_df = picard_metrics_utils.parse_cvg_metrics(picard_file)[2]
    assert picard_metrics_utils.parse_cvg_metrics(picard_file)[0] == "AlignmentSummaryMetrics"
    assert metric_df.TOTAL_READS[0] == 640936116
    assert histogram_df.READ_LENGTH[4] == 29