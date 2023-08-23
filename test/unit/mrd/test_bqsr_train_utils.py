import filecmp
from os.path import join as pjoin
from test import get_resource_dir, test_dir

from ugvc.mrd.bqsr_train_utils import prepare_featuremap_for_model

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)


def test_prepare_featuremap_for_training_no_test(tmpdir):
    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    expected_training_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.intersect.training.downsampled.vcf.gz",
    )
    downsampled_training_featuremap_vcf, _ = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        training_set_size=12,
        balance_motifs=False,
        random_seed=0,
    )
    assert filecmp.cmp(downsampled_training_featuremap_vcf, expected_training_featuremap_vcf)


def test_prepare_featuremap_for_training_with_test(tmpdir):
    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    expected_training_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.intersect.training.downsampled.2.vcf.gz",
    )
    expected_test_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.intersect.test.downsampled.2.vcf.gz",
    )
    (downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf,) = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        training_set_size=12,
        test_set_size=3,
        balance_motifs=False,
        random_seed=0,
    )
    assert filecmp.cmp(downsampled_training_featuremap_vcf, expected_training_featuremap_vcf)
    assert filecmp.cmp(downsampled_test_featuremap_vcf, expected_test_featuremap_vcf)
