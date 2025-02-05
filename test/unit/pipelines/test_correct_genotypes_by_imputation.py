import numpy as np

from ugvc.pipelines import correct_genotypes_by_imputation as corgen
from ugbio_core.vcfbed.vcftools import genotype_ordering


class TestCorrectGenotypesByImputation:
    def test_convert_ds_to_genotype_imputation_priors(self):
        # Homozygous biallelic variant
        ds_ar = np.array([2])
        num_alt = 1
        gt_ar = genotype_ordering(num_alt)
        epsilon = 0.01
        assert np.all(
            corgen._convert_ds_to_genotype_imputation_priors(ds_ar, gt_ar, num_alt, epsilon)
            == np.array([1, 0.01, 0.99])
        )
        # Heterozygous biallelic variant
        ds_ar = np.array([1])
        num_alt = 1
        gt_ar = genotype_ordering(num_alt)
        epsilon = 0.01
        assert np.all(
            corgen._convert_ds_to_genotype_imputation_priors(ds_ar, gt_ar, num_alt, epsilon)
            == np.array([1, 0.99, 0.01])
        )
        # Heterozygous tri-allelic variant
        ds_ar = np.array([1, 1])
        num_alt = 2
        gt_ar = genotype_ordering(num_alt)
        epsilon = 0.01
        assert np.all(
            corgen._convert_ds_to_genotype_imputation_priors(ds_ar, gt_ar, num_alt, epsilon)
            == np.array([1, 0.99, 0.01, 0.99, 0.99, 0.01])
        )
        # Heterozygous tri-allelic variant with missing imputation data
        ds_ar = np.array([2, None], dtype=float)
        num_alt = 2
        gt_ar = genotype_ordering(num_alt)
        epsilon = 0.01
        assert np.all(
            corgen._convert_ds_to_genotype_imputation_priors(ds_ar, gt_ar, num_alt, epsilon)
            == np.array([1, 0.01, 0.99, 0.01, 0.01, 0.01])
        )
