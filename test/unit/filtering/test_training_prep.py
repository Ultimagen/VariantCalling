import os.path
import pathlib
from test import get_resource_dir

import pandas as pd

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

    def test_label_with_approximate_gt(self, tmpdir):
        inputs_dir = self.inputs_dir
        inputs_file = str(pathlib.Path(inputs_dir, "006919_no_frd_chr1_1_5000000.vcf.gz"))
        blacklist_file = str(pathlib.Path(inputs_dir, "blacklist_chr1_1_5000000.h5"))
        tprep.label_with_approximate_gt(
            inputs_file,
            blacklist_file,
            chromosomes_to_read=["chr1"],
            output_file=str(pathlib.Path(tmpdir, "output.h5")),
        )
        assert os.path.exists(str(pathlib.Path(tmpdir, "output.h5")))
        vc = pd.read_hdf(str(pathlib.Path(tmpdir, "output.h5")), key="chr1")["label"].value_counts()
        assert len(vc) == 2
        assert vc[1] == 8715
        assert vc[0] == 2003
