import os
import unittest
from os.path import dirname
import filecmp
import sys

import python.pipelines.pathmagic
sys.path.append(dirname(python.pipelines.pathmagic.__file__))

from python.pipelines import evaluate_concordance
from test import get_resource_dir, make_test_outputs_dir


class TestEvaluateConcordance(unittest.TestCase):
    inputs_dir = get_resource_dir(__file__)
    test_outputs_dir = make_test_outputs_dir(__file__)

    def test_evaluate_concordance(self):
        input_file = f'{self.inputs_dir}/test.untrained.h5'
        output_prefix = f'{self.test_outputs_dir}/test'
        os.makedirs(dirname(output_prefix), exist_ok=True)

        evaluate_concordance.run(
            ['--input_file', input_file,
             '--output_prefix', f'{self.test_outputs_dir}/test'
             ])

        expected_stats_file = f'{self.inputs_dir}/expected.out.stats.csv'
        out_stats_file = f'{self.test_outputs_dir}/test.stats.csv'
        self.assertTrue(filecmp.cmp(expected_stats_file, out_stats_file, shallow=False),
                        'stats files are not identical')
