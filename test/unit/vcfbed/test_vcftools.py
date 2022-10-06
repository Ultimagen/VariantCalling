from os.path import join as pjoin
from test import get_resource_dir

import pandas as pd

import ugvc.vcfbed.vcftools as vcftools

inputs_dir = get_resource_dir(__file__)


def test_bed_files_output():
    # snp_fp testing
    data = pd.read_hdf(pjoin(inputs_dir, "BC10.chr1.h5"), key="concordance")
    snp_fp = vcftools.FilterWrapper(data).get_snp().get_fp().get_df()
    assert all([x is False for x in snp_fp["indel"]])
    assert all([x == "fp" for x in snp_fp["classify"]])

    # snp_fn testing
    snp_fn = vcftools.FilterWrapper(data).get_snp().get_fn().get_df()
    assert all([x is False for x in snp_fn["indel"]])
    assert all(
        [
            row["classify"] == "fn"
            or (
                row["classify"] == "tp"
                and (row["filter"] == "LOW_SCORE")
                and (row["filter"] != "PASS")
            )
            for index, row in snp_fn.iterrows()
        ]
    )

    # hmer
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer().get_df()
    assert all(hmer_fn["indel"])
    assert all([x > 0 for x in hmer_fn["hmer_indel_length"]])
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer(val_start=1, val_end=1).get_df()
    assert all(hmer_fn["indel"])
    assert all([x == 1 for x in hmer_fn["hmer_indel_length"]])
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer(val_start=3, val_end=10).get_df()
    assert all(hmer_fn["indel"])
    assert all([x >= 3 & x <= 10 for x in hmer_fn["hmer_indel_length"]])

    # hmer_fp testing
    hmer_fp = vcftools.FilterWrapper(data).get_h_mer().get_fp().get_df()
    assert all(hmer_fp["indel"])
    assert all([x > 0 for x in hmer_fp["hmer_indel_length"]])
    assert all([x == "fp" for x in hmer_fp["classify"]])

    # hmer_fn testing
    hmer_fn = vcftools.FilterWrapper(data).get_h_mer().get_fn().get_df()
    assert all(hmer_fn["indel"])
    assert all([x > 0 for x in hmer_fn["hmer_indel_length"]])
    assert all(
        [
            row["classify"] == "fn"
            or (
                row["classify"] == "tp"
                and (row["filter"] == "LOW_SCORE")
                and (row["filter"] != "PASS")
            )
            for index, row in hmer_fn.iterrows()
        ]
    )

    # non_hmer_fp testing
    non_hmer_fp = vcftools.FilterWrapper(data).get_non_h_mer().get_fp().get_df()
    assert all(non_hmer_fp["indel"])
    assert all([x == 0 for x in non_hmer_fp["hmer_indel_length"]])
    assert all([x == "fp" for x in non_hmer_fp["classify"]])

    # non_hmer_fn testing
    non_hmer_fn = vcftools.FilterWrapper(data).get_non_h_mer().get_fn().get_df()
    assert all(non_hmer_fn["indel"])
    assert all([x == 0 for x in non_hmer_fn["hmer_indel_length"]])
    assert all(
        [
            row["classify"] == "fn"
            or (
                row["classify"] == "tp"
                and (row["filter"] == "LOW_SCORE")
                and (row["filter"] != "PASS")
            )
            for index, row in non_hmer_fn.iterrows()
        ]
    )


def test_bed_output_when_no_tree_score():  # testing the case when there is no tree_score and there is blacklist
    data = pd.read_hdf(pjoin(inputs_dir, "exome.h5"), key="concordance")
    df = vcftools.FilterWrapper(data)
    result = dict(df.get_fn().bed_format(kind="fn").get_df()["itemRgb"].value_counts())
    expected_result = {
        vcftools.FilteringColors.BLACKLIST.value: 169,
        vcftools.FilteringColors.CLEAR.value: 89,
        vcftools.FilteringColors.BORDERLINE.value: 39,
    }
    for k in result:
        assert result[k] == expected_result[k]

    df = vcftools.FilterWrapper(data)
    # since there is no tree_score all false positives should be the same color
    result = dict(df.get_fp().bed_format(kind="fp").get_df()["itemRgb"].value_counts())

    assert len(result.keys()) == 1


def test_get_region_around_variant():
    vpos = 100
    vlocs = []
    assert vcftools.get_region_around_variant(vpos, vlocs, 10) == (95, 105)


class TestGetVcfDf:
    def test_get_vcf_df(self):
        input_vcf = pjoin(inputs_dir, "test_get_vcf_df.vcf.gz")
        df = vcftools.get_vcf_df(input_vcf)
        non_nan_columns = list(df.dropna(axis=1, how="all").columns)
        non_nan_columns.sort()
        assert non_nan_columns == [
            "ac",
            "ad",
            "af",
            "alleles",
            "an",
            "baseqranksum",
            "chrom",
            "db",
            "dp",
            "excesshet",
            "filter",
            "fs",
            "gnomad_af",
            "gq",
            "gt",
            "id",
            "indel",
            "mleac",
            "mleaf",
            "mq",
            "mqranksum",
            "pl",
            "pos",
            "qd",
            "qual",
            "readposranksum",
            "ref",
            "sor",
            "tree_score",
            "variant_type",
            "x_css",
            "x_gcc",
            "x_ic",
            "x_il",
            "x_lm",
            "x_rm",
        ]

    def test_get_vcf_df_use_qual(self):
        input_vcf = pjoin(inputs_dir, "test_get_vcf_df.vcf.gz")
        df = vcftools.get_vcf_df(input_vcf, scoring_field="QUAL")
        assert all(df["qual"] == df["tree_score"])

    def test_get_vcf_df_ignore_fields(self):
        input_vcf = pjoin(inputs_dir, "test_get_vcf_df.vcf.gz")
        ignore_fields = ["x_css", "x_gcc", "x_ic", "x_il", "x_lm", "x_rm"]
        df = vcftools.get_vcf_df(input_vcf, ignore_fields=ignore_fields)
        non_nan_columns = list(df.dropna(axis=1, how="all").columns)
        non_nan_columns.sort()
        assert non_nan_columns == [
            "ac",
            "ad",
            "af",
            "alleles",
            "an",
            "baseqranksum",
            "chrom",
            "db",
            "dp",
            "excesshet",
            "filter",
            "fs",
            "gnomad_af",
            "gq",
            "gt",
            "id",
            "indel",
            "mleac",
            "mleaf",
            "mq",
            "mqranksum",
            "pl",
            "pos",
            "qd",
            "qual",
            "readposranksum",
            "ref",
            "sor",
            "tree_score",
            "variant_type",
        ]
        for x in ignore_fields:
            assert x not in df.columns
