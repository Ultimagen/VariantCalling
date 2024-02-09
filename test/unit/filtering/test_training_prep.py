import pathlib
from test import get_resource_dir

import pandas as pd

import ugvc.filtering.training_prep as tprep


def test_calculate_labeled_vcf():
    inputs_dir = get_resource_dir(__file__)
    inputs_file = str(pathlib.Path(inputs_dir, "input.vcf.gz"))
    vcfeval_output = str(pathlib.Path(inputs_dir, "vcfeval_output.vcf.gz"))
    expected_result_file = str(pathlib.Path(inputs_dir, "expected_result_calculate_labeled_vcf.h5"))
    expected_results = pd.DataFrame(pd.read_hdf(expected_result_file, key="result"))
    labeled_df = tprep.calculate_labeled_vcf(inputs_file, vcfeval_output, contig="chr20")
    pd.testing.assert_frame_equal(labeled_df, expected_results)
