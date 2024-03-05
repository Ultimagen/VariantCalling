import pathlib
from test import get_resource_dir

import pandas as pd

import ugvc.filtering.training_prep as tprep


def test_prepare_ground_truth(mocker, tmpdir):
    inputs_dir = get_resource_dir(__file__)
    calls_file = str(pathlib.Path(inputs_dir, "input.vcf.gz"))
    vcfeval_output_file = pathlib.Path(inputs_dir, "vcfeval_output.vcf.gz")
    # bypass well-tested run_vcfeval_concordance
    mocker.patch(
        "ugvc.comparison.vcf_pipeline_utils.VcfPipelineUtils.run_vcfeval_concordance", return_value=vcfeval_output_file
    )
    joint_df = pd.read_hdf(str(pathlib.Path(inputs_dir, "labeled_df.h5")), key="df")
    # bypass creating of the joint vcf (good unit test)
    mocker.patch("ugvc.filtering.training_prep.calculate_labeled_vcf", return_value=joint_df)
    reference_path = str(pathlib.Path(inputs_dir, "ref_fragment.fa.gz"))
    output_file = str(pathlib.Path(tmpdir, "output.h5"))
    tprep.prepare_ground_truth(calls_file, "", "", reference_path, output_file, chromosome=["chr21"])
    assert pathlib.Path(output_file).exists()
    output_df = pd.DataFrame(pd.read_hdf(output_file, key="chr21"))
    expected_df = pd.DataFrame(pd.read_hdf(str(pathlib.Path(inputs_dir, "expected_output.h5")), key="chr21"))
    pd.testing.assert_frame_equal(
        output_df.drop(["spanning_deletion", "multiallelic_group"], axis=1),
        expected_df.drop(["spanning_deletion", "multiallelic_group"], axis=1),
    )
