import tempfile
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
from pandas.testing import assert_frame_equal

from ugvc.scripts.sorter_to_h5 import sorter_to_h5

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
report_path = pjoin(test_dir, "..", "ugvc", "reports")
base_file_name = "026532-Lb_1866-Z0058-CATCTCAGTGCAATGAT"


def test_sorter_to_h5():
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_h5 = sorter_to_h5(
            input_csv_file=pjoin(inputs_dir, base_file_name + ".csv"),
            input_json_file=pjoin(inputs_dir, base_file_name + ".json"),
            metric_mapping_file=pjoin(report_path, "sorter_output_to_aggregated_metrics_h5.csv"),
            output_dir=tmpdirname,
        )
        expected_output_file = pjoin(inputs_dir, base_file_name + ".aggregated_metrics.h5")
        with pd.HDFStore(output_h5) as hdf:
            output_h5_keys = hdf.keys()
        with pd.HDFStore(expected_output_file) as hdf:
            expected_output_file_keys = hdf.keys()
        assert output_h5_keys == expected_output_file_keys
        for key in output_h5_keys:
            assert_frame_equal(pd.read_hdf(output_h5, key), pd.read_hdf(expected_output_file, key))
