import pathlib
from test import get_resource_dir

import pandas as pd
import pytest

import ugvc.filtering.training_prep as tprep


class TestTrainingPrep:
    inputs_dir = get_resource_dir(__file__)

    def test_calculate_labeled_vcf(self):
        inputs_dir = self.inputs_dir
        inputs_file = str(pathlib.Path(inputs_dir, "input.vcf.gz"))
        vcfeval_output = str(pathlib.Path(inputs_dir, "vcfeval_output.vcf.gz"))
        expected_result_file = str(pathlib.Path(inputs_dir, "expected_result_calculate_labeled_vcf.h5"))
        expected_results = pd.DataFrame(pd.read_hdf(expected_result_file, key="result"))
        labeled_df = tprep.calculate_labeled_vcf(inputs_file, vcfeval_output, contig="chr20")
        pd.testing.assert_frame_equal(labeled_df, expected_results)

    def test_calculate_labels(self):
        inputs_dir = self.inputs_dir
        joint_vcf_df_file = str(pathlib.Path(inputs_dir, "expected_result_calculate_labeled_vcf.h5"))
        joint_vcf_df = pd.DataFrame(pd.read_hdf(joint_vcf_df_file, key="result"))
        labeled_df = tprep.calculate_labels(joint_vcf_df)
        expected_result_file = str(pathlib.Path(inputs_dir, "expected_labels.h5"))
        pd.testing.assert_series_equal(labeled_df, pd.read_hdf(expected_result_file, key="labels"))  # type: ignore

    def test_encode_labels(self):
        labels = [(0, 1), (0, 0), (1, 0), (1, 1)]
        encoded_labels = tprep.encode_labels(labels)
        expected_result = [0, 2, 0, 1]
        assert encoded_labels == expected_result
        with pytest.raises(ValueError):
            tprep.encode_labels([(0, 1, 2), (0, 0, 1)])  # type: ignore
        with pytest.raises(ValueError):
            tprep.encode_labels([(0, 2), (1, 2)])
