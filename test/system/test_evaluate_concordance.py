import filecmp
import os
from os.path import dirname
from test import get_resource_dir

from ugvc.pipelines import evaluate_concordance


class TestEvaluateConcordance:
    inputs_dir = get_resource_dir(__file__)

    def test_evaluate_concordance(self, tmpdir):
        input_file = f"{self.inputs_dir}/test.untrained.h5"
        output_prefix = f"{tmpdir}/test"
        os.makedirs(dirname(output_prefix), exist_ok=True)

        evaluate_concordance.run(["--input_file", input_file, "--output_prefix", f"{tmpdir}/test"])

        expected_stats_file = f"{self.inputs_dir}/expected.out.stats.csv"
        out_stats_file = f"{tmpdir}/test.stats.csv"
        assert filecmp.cmp(expected_stats_file, out_stats_file, shallow=False), "stats files are not identical"
