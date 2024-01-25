import os
import subprocess
from collections import defaultdict
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np
import pysam
from scipy.stats import ttest_1samp

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
    downsampled_training_featuremap_vcf, _, _, _, _ = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        train_set_size=12,
        test_set_size=3,
        balanced_sampling_info_fields=None,
        random_seed=0,
    )

    # Since we use random downsampling the train_set_size might differ slightly from expected
    n_variants = __count_variants(downsampled_training_featuremap_vcf)
    assert n_variants >= 8 and n_variants <= 16


def test_prepare_featuremap_for_model_with_prefilter(tmpdir):
    """Test that downsampling training-set works as expected"""

    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    pre_filter_bcftools_include = "(X_SCORE>4) && (X_EDIST<10)"
    (downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf, _, _, _) = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        train_set_size=100,
        test_set_size=100,
        balanced_sampling_info_fields=None,
        pre_filter_bcftools_include=pre_filter_bcftools_include,
        random_seed=0,
    )
    # In this scenario we are pre-filtering the test data so that only 4 FeatureMap entries pass:
    total_variants = int(
        subprocess.check_output(
            f"bcftools view -H {input_featuremap_vcf} -i '{pre_filter_bcftools_include}' | wc -l",
            shell=True,
        )
        .decode()
        .strip()
    )
    assert total_variants == 4
    # and since we are asking for more entries than are available, we should get all of them in equal ratios
    n_variants = __count_variants(downsampled_training_featuremap_vcf)
    assert n_variants == 2
    n_variants = __count_variants(downsampled_test_featuremap_vcf)
    assert n_variants == 2


def test_prepare_featuremap_for_model_with_motif_balancing(tmpdir):
    """Test that downsampling training-set works as expected"""

    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_LuNgs_08.annotated_featuremap.vcf.gz",
    )
    balanced_sampling_info_fields = ["trinuc_context", "is_forward"]
    train_set_size = (4**3) * 10  # 10 variants per context
    for random_seed in range(2):
        downsampled_training_featuremap_vcf, _, _, _, _ = prepare_featuremap_for_model(
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
        # T-test that the number of variants per context is in line with a uniform with to 99% confidence
        _, pvalue = ttest_1samp(
            list(balanced_sampling_info_fields_counter.values()),
            np.mean(list(balanced_sampling_info_fields_counter.values())),
        )
        assert pvalue > 0.01
        os.remove(downsampled_training_featuremap_vcf)
        os.remove(downsampled_training_featuremap_vcf + ".tbi")


def test_prepare_featuremap_for_model_training_and_test_sets(tmpdir):
    """Test that downsampling of training and test sets works as expected"""
    input_featuremap_vcf = pjoin(
        inputs_dir,
        "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz",
    )
    (downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf, _, _, _) = prepare_featuremap_for_model(
        workdir=tmpdir,
        input_featuremap_vcf=input_featuremap_vcf,
        train_set_size=12,
        test_set_size=3,
        balanced_sampling_info_fields=None,
        random_seed=0,
    )
    assert __count_variants(downsampled_training_featuremap_vcf) == 12
    assert __count_variants(downsampled_test_featuremap_vcf) == 3
