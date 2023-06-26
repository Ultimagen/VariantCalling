import filecmp
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd

from ugvc.mrd.bqsr_train_utils import prepare_featuremap_for_training

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)


def test_prepare_featuremap_for_training(tmpdir):
    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    expected_output_featuremap_vcf = pjoin(
        inputs_dir, "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.intersect.downsampled.vcf.gz"
    )
    downsampled_featuremap_vcf, _ = prepare_featuremap_for_training(
        workdir=tmpdir, input_featuremap_vcf=input_featuremap_vcf, training_set_size=12
    )
    assert filecmp.cmp(downsampled_featuremap_vcf, expected_output_featuremap_vcf)
