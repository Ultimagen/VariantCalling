import filecmp
from os.path import basename, exists
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np

from ugvc.utils import metrics_utils

inputs_dir = get_resource_dir(__file__)


def test_preprocess_h5_key_with_slash():
    assert metrics_utils.preprocess_h5_key("/foo") == "foo"


def test_preprocess_h5_key_without_slash():
    assert metrics_utils.preprocess_h5_key("foo") == "foo"


def test_should_skip_h5_key_true():
    assert metrics_utils.should_skip_h5_key("str_histogram_123str", "histogram")


def test_should_skip_h5_key_false():
    assert not metrics_utils.should_skip_h5_key("str_his_togram_123str", "histogram")


def test_get_h5_keys():
    metrics_h5_path = pjoin(inputs_dir, "140479-BC21_aggregated_metrics.h5")
    assert np.array_equal(
        metrics_utils.get_h5_keys(metrics_h5_path),
        [
            "/AlignmentSummaryMetrics",
            "/DuplicationMetrics",
            "/GcBiasDetailMetrics",
            "/GcBiasSummaryMetrics",
            "/QualityYieldMetrics",
            "/RawWgsMetrics",
            "/WgsMetrics",
            "/histogram_AlignmentSummaryMetrics",
            "/histogram_RawWgsMetrics",
            "/histogram_WgsMetrics",
            "/histogram_coverage",
            "/stats_coverage",
        ],
    )


def test_convert_h5_to_json():
    metrics_h5_path = pjoin(inputs_dir, "140479-BC21_aggregated_metrics.h5")
    metrics_json_path = pjoin(inputs_dir, "140479-BC21_aggregated_metrics.json")
    with open(metrics_json_path, "r") as json_file:
        data = json_file.read()
    assert f'{metrics_utils.convert_h5_to_json(metrics_h5_path, "metrics", "histogram")}\n' == data


picard_file = pjoin(inputs_dir, "140479-BC21.alignment_summary_metrics")


def test_parse_cvg_metrics():
    metric_df = metrics_utils.parse_cvg_metrics(picard_file)[1]
    histogram_df = metrics_utils.parse_cvg_metrics(picard_file)[2]
    assert metrics_utils.parse_cvg_metrics(picard_file)[0] == "AlignmentSummaryMetrics"
    assert metric_df.TOTAL_READS[0] == 640936116
    assert histogram_df.READ_LENGTH[4] == 29


def test_merge_trimmer_histograms(tmpdir):
    suf = "A_hmer_start.T_hmer_start.A_hmer_end.T_hmer_end.native_adapter_with_leading_C.histogram.csv"
    trimmer_histograms = [
        pjoin(
            inputs_dir,
            f"029917001_1_Z0098-{suf}",
        ),
        pjoin(
            inputs_dir,
            f"029917001_2_Z0098-{suf}",
        ),
    ]
    expected_output = pjoin(
        inputs_dir,
        f"EXPECTED.029917001_Z0098-{suf}",
    )
    output_path = pjoin(
        tmpdir,
        basename(trimmer_histograms[0]),
    )
    merged_histogram = metrics_utils.merge_trimmer_histograms(trimmer_histograms=trimmer_histograms, output_path=tmpdir)
    assert exists(merged_histogram)
    assert merged_histogram == output_path
    assert filecmp.cmp(merged_histogram, expected_output)

    output_path = pjoin(
        tmpdir,
        basename(trimmer_histograms[1]),
    )
    merged_histogram = metrics_utils.merge_trimmer_histograms(
        trimmer_histograms=trimmer_histograms[1], output_path=tmpdir
    )
    assert exists(merged_histogram)
    assert filecmp.cmp(merged_histogram, trimmer_histograms[1])
