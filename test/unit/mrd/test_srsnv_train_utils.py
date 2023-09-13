import os
from collections import defaultdict
from os.path import join as pjoin
from test import get_resource_dir

import pysam

from ugvc.mrd.srsnv_training_utils import prepare_featuremap_for_model

inputs_dir = get_resource_dir(__file__)


def __count_variants(vcf_file):
    counter = 0
    for _ in pysam.VariantFile(vcf_file):
        counter += 1
    return counter


def test_prepare_featuremap_for_model(tmpdir):
    """Test that downsampling training-set works as expected"""

    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    downsampled_training_featuremap_vcf, _, _ = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        train_set_size=12,
        balanced_sampling_info_fields=None,
        random_seed=0,
    )

    # Since we use random downsampling the train_set_size might differ slightly from expected
    assert __count_variants(downsampled_training_featuremap_vcf) == 11


def test_prepare_featuremap_for_model_with_motif_balancing(tmpdir):
    """Test that downsampling training-set works as expected"""

    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_LuNgs_08.annotated_featuremap.vcf.gz",
    )
    balanced_sampling_info_fields = ["trinuc_context", "is_forward"]
    train_set_size = (4**3) * 10  # 10 variants per context
    for random_seed in range(2):
        downsampled_training_featuremap_vcf, _, _ = prepare_featuremap_for_model(
            workdir=tmpdir,
            input_featuremap_vcf=input_featuremap_vcf,
            train_set_size=train_set_size,
            test_set_size=train_set_size,
            balanced_sampling_info_fields=balanced_sampling_info_fields,
            random_seed=random_seed,
        )
        assert __count_variants(downsampled_training_featuremap_vcf) == train_set_size

        balanced_sampling_info_fields_counter = defaultdict(int)
        with pysam.VariantFile(downsampled_training_featuremap_vcf) as fmap:
            for record in fmap.fetch():
                balanced_sampling_info_fields_counter[
                    tuple(record.info.get(info_field) for info_field in balanced_sampling_info_fields)
                ] += 1
        assert sum(balanced_sampling_info_fields_counter.values()) == train_set_size
        os.remove(downsampled_training_featuremap_vcf)
        os.remove(downsampled_training_featuremap_vcf + ".tbi")


def test_prepare_featuremap_for_model_training_and_test_sets(tmpdir):
    """Test that downsampling of training and test sets works as expected"""
    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    (downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf, _,) = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        train_set_size=12,
        test_set_size=3,
        balanced_sampling_info_fields=False,
        random_seed=0,
    )
    assert __count_variants(downsampled_training_featuremap_vcf) == 12
    assert __count_variants(downsampled_test_featuremap_vcf) == 3
