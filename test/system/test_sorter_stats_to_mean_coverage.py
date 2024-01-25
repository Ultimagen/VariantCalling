from os.path import join as pjoin
from test import get_resource_dir

from ugvc.pipelines.mrd.sorter_stats_to_mean_coverage import run, sorter_stats_to_mean_coverage


def test_sorter_stats_to_mean_coverage(tmpdir):
    inputs_dir = get_resource_dir(__file__)
    input_json = pjoin(inputs_dir, "Pa_394_Plasma.Lb_1597.runs_021144_021150_023049.json")
    expected_coverage = 196
    output_file = pjoin(tmpdir, "mean_coverage.txt")
    sorter_stats_to_mean_coverage(input_json, output_file)
    with open(output_file) as f:
        assert f.read() == f"{expected_coverage} "


def test_run_sorter_stats_to_mean_coverage(tmpdir):
    inputs_dir = get_resource_dir(__file__)
    input_json = pjoin(inputs_dir, "Pa_394_Plasma.Lb_1597.runs_021144_021150_023049.json")
    expected_coverage = 196
    output_file = pjoin(tmpdir, "mean_coverage.txt")
    run(
        [
            "sorter_stats_to_mean_coverage",
            "-i",
            input_json,
            "-o",
            output_file,
        ]
    )
    with open(output_file) as f:
        assert f.read() == f"{expected_coverage} "
