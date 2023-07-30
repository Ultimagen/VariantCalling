import tempfile
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ugvc.mrd.balanced_strand_utils import (
    add_strand_ratios_and_categories_to_featuremap,
    plot_balanced_strand_ratio,
    plot_strand_ratio_category,
    plot_strand_ratio_category_concordnace,
    read_balanced_strand_trimmer_histogram,
)

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
input_histogram_LAv5and6_csv = pjoin(
    inputs_dir,
    "130713_UGAv3-51.trimming.A_hmer_5.T_hmer_5.A_hmer_3.T_hmer_3.native_adapter_with_leading_C.histogram.csv",
)
parsed_histogram_LAv5and6_parquet = pjoin(inputs_dir, "130713_UGAv3-51.parsed_histogram.parquet")
input_histogram_LAv5_csv = pjoin(inputs_dir, "130715_UGAv3-132.trimming.A_hmer.T_hmer.histogram.csv")
parsed_histogram_LAv5_parquet = pjoin(inputs_dir, "130715_UGAv3-132.parsed_histogram.parquet")
input_featuremap_LAv5and6 = pjoin(inputs_dir, "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz")
expected_output_featuremap_LAv5and6 = pjoin(
    inputs_dir,
    "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.with_strand_ratios.vcf.gz",
)


def _assert_files_are_identical(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        assert f1.read() == f2.read()


def test_read_balanced_strand_LAv5and6_trimmer_histogram():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.parquet")
        df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
            input_histogram_LAv5and6_csv, output_filename=tmp_out_path
        )
        df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
        df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
        assert_frame_equal(
            df_trimmer_histogram,
            df_trimmer_histogram_expected,
        )
        assert_frame_equal(
            df_trimmer_histogram_from_parquet,
            df_trimmer_histogram_expected,
        )


def test_read_balanced_strand_LAv5_trimmer_histogram():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.parquet")
        df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
            input_histogram_LAv5_csv, output_filename=tmp_out_path
        )
        df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
        df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
        assert_frame_equal(
            df_trimmer_histogram,
            df_trimmer_histogram_expected,
        )
        assert_frame_equal(
            df_trimmer_histogram_from_parquet,
            df_trimmer_histogram_expected,
        )


def test_plot_balanced_strand_ratio():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.png")
        df_trimmer_histogram_LAv5and6_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
        plot_balanced_strand_ratio(
            df_trimmer_histogram_LAv5and6_expected,
            output_filename=tmp_out_path,
            title="test",
        )
        df_trimmer_histogram_LAv5_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
        plot_balanced_strand_ratio(
            df_trimmer_histogram_LAv5_expected,
            output_filename=tmp_out_path,
            title="test",
        )


def test_plot_strand_ratio_category():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.png")
        df_trimmer_histogram_LAv5and6_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
        plot_strand_ratio_category(
            df_trimmer_histogram_LAv5and6_expected,
            title="test",
            output_filename=tmp_out_path,
            ax=None,
        )
        df_trimmer_histogram_LAv5_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
        plot_strand_ratio_category(
            df_trimmer_histogram_LAv5_expected,
            title="test",
            output_filename=tmp_out_path,
            ax=None,
        )


def test_add_strand_ratios_and_categories_to_featuremap():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.vcf.gz")
        add_strand_ratios_and_categories_to_featuremap(
            input_featuremap_vcf=input_featuremap_LAv5and6,
            output_featuremap_vcf=tmp_out_path,
        )
        _assert_files_are_identical(
            expected_output_featuremap_LAv5and6,
            tmp_out_path,
        )


def test_plot_strand_ratio_category_concordnace():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.png")
        df_trimmer_histogram_LAv5and6_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
        plot_strand_ratio_category_concordnace(
            df_trimmer_histogram_LAv5and6_expected,
            title="test",
            output_filename=tmp_out_path,
            axs=None,
        )
        df_trimmer_histogram_LAv5_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
        with pytest.raises(ValueError):
            plot_strand_ratio_category_concordnace(
                df_trimmer_histogram_LAv5_expected,
                title="test",
                output_filename=tmp_out_path,
                axs=None,
            )
