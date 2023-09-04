from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ugvc.mrd.balanced_strand_utils import (
    BalancedStrandAdapterVersions,
    add_strand_ratios_and_categories_to_featuremap,
    balanced_strand_analysis,
    collect_statistics,
    plot_balanced_strand_ratio,
    plot_strand_ratio_category,
    plot_strand_ratio_category_concordnace,
    plot_trimmer_histogram,
    read_balanced_strand_trimmer_histogram,
)

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
input_histogram_LAv5and6_csv = pjoin(
    inputs_dir,
    "130713_UGAv3-51.trimming.A_hmer_5.T_hmer_5.A_hmer_3.T_hmer_3.native_adapter_with_leading_C.histogram.csv",
)
parsed_histogram_LAv5and6_parquet = pjoin(inputs_dir, "130713_UGAv3-51.parsed_histogram.parquet")
sorter_stats_LAv5and6_csv = pjoin(inputs_dir, "130713-UGAv3-51.sorter_stats.csv")
collected_stats_LAv5and6_h5 = pjoin(inputs_dir, "130713-UGAv3-51.stats.h5")
input_histogram_LAv5_csv = pjoin(inputs_dir, "130715_UGAv3-132.trimming.A_hmer.T_hmer.histogram.csv")
parsed_histogram_LAv5_parquet = pjoin(inputs_dir, "130715_UGAv3-132.parsed_histogram.parquet")
sorter_stats_LAv5_csv = pjoin(inputs_dir, "130715-UGAv3-51.sorter_stats.csv")
collected_stats_LAv5_h5 = pjoin(inputs_dir, "130715-UGAv3-51.stats.h5")
input_featuremap_LAv5and6 = pjoin(inputs_dir, "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.vcf.gz")
expected_output_featuremap_LAv5and6 = pjoin(
    inputs_dir,
    "333_CRCs_39_LAv5and6.featuremap.single_substitutions.subsample.with_strand_ratios.vcf.gz",
)


def _assert_files_are_identical(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        assert f1.read() == f2.read()


def test_read_balanced_strand_LAv5and6_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
        BalancedStrandAdapterVersions.LA_v5and6,
        input_histogram_LAv5and6_csv,
        output_filename=tmp_out_path,
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


def test_read_balanced_strand_LAv5_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
        BalancedStrandAdapterVersions.LA_v5and6,
        input_histogram_LAv5_csv,
        output_filename=tmp_out_path,
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


def test_plot_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
        BalancedStrandAdapterVersions.LA_v5,
        input_histogram_LAv5_csv,
        output_filename=tmp_out_path,
    )
    plot_trimmer_histogram(
        BalancedStrandAdapterVersions.LA_v5,
        df_trimmer_histogram,
        output_filename=tmp_out_path,
    )
    df_trimmer_histogram = read_balanced_strand_trimmer_histogram(
        BalancedStrandAdapterVersions.LA_v5and6,
        input_histogram_LAv5and6_csv,
        output_filename=tmp_out_path,
    )
    plot_trimmer_histogram(
        BalancedStrandAdapterVersions.LA_v5and6,
        df_trimmer_histogram,
        output_filename=tmp_out_path,
    )


def test_collect_statistics(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.h5")
    collect_statistics(
        BalancedStrandAdapterVersions.LA_v5and6,
        trimmer_histogram_csv=input_histogram_LAv5and6_csv,
        sorter_stats_csv=sorter_stats_LAv5and6_csv,
        output_filename=tmp_out_path,
    )
    f1 = collected_stats_LAv5and6_h5
    f2 = tmp_out_path
    with pd.HDFStore(f1) as fh1, pd.HDFStore(f2) as fh2:
        assert fh1.keys() == fh2.keys()
        keys = fh1.keys()
    for k in keys:
        assert_frame_equal(
            pd.read_hdf(f1, k),
            pd.read_hdf(f2, k),
        )

    collect_statistics(
        BalancedStrandAdapterVersions.LA_v5,
        trimmer_histogram_csv=input_histogram_LAv5_csv,
        sorter_stats_csv=sorter_stats_LAv5and6_csv,
        output_filename=tmp_out_path,
    )
    f1 = collected_stats_LAv5_h5
    f2 = tmp_out_path
    with pd.HDFStore(f1) as fh1, pd.HDFStore(f2) as fh2:
        assert fh1.keys() == fh2.keys()
        keys = fh1.keys()
    for k in keys:
        assert_frame_equal(
            pd.read_hdf(f1, k),
            pd.read_hdf(f2, k),
        )


def test_plot_balanced_strand_ratio(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_LAv5and6_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
    plot_balanced_strand_ratio(
        BalancedStrandAdapterVersions.LA_v5and6,
        df_trimmer_histogram_LAv5and6_expected,
        output_filename=tmp_out_path,
        title="test",
    )
    df_trimmer_histogram_LAv5_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
    plot_balanced_strand_ratio(
        BalancedStrandAdapterVersions.LA_v5,
        df_trimmer_histogram_LAv5_expected,
        output_filename=tmp_out_path,
        title="test",
    )


def test_plot_strand_ratio_category(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_LAv5and6_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
    plot_strand_ratio_category(
        BalancedStrandAdapterVersions.LA_v5and6,
        df_trimmer_histogram_LAv5and6_expected,
        title="test",
        output_filename=tmp_out_path,
        ax=None,
    )
    df_trimmer_histogram_LAv5_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
    plot_strand_ratio_category(
        BalancedStrandAdapterVersions.LA_v5,
        df_trimmer_histogram_LAv5_expected,
        title="test",
        output_filename=tmp_out_path,
        ax=None,
    )


def test_add_strand_ratios_and_categories_to_featuremap(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.vcf.gz")
    add_strand_ratios_and_categories_to_featuremap(
        BalancedStrandAdapterVersions.LA_v5and6,
        input_featuremap_vcf=input_featuremap_LAv5and6,
        output_featuremap_vcf=tmp_out_path,
    )
    _assert_files_are_identical(
        expected_output_featuremap_LAv5and6,
        tmp_out_path,
    )


def test_plot_strand_ratio_category_concordnace(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_LAv5and6_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
    plot_strand_ratio_category_concordnace(
        BalancedStrandAdapterVersions.LA_v5and6,
        df_trimmer_histogram_LAv5and6_expected,
        title="test",
        output_filename=tmp_out_path,
        axs=None,
    )
    df_trimmer_histogram_LAv5_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
    with pytest.raises(ValueError):
        plot_strand_ratio_category_concordnace(
            BalancedStrandAdapterVersions.LA_v5,
            df_trimmer_histogram_LAv5_expected,
            title="test",
            output_filename=tmp_out_path,
            axs=None,
        )


def test_balanced_strand_analysis(tmpdir):
    balanced_strand_analysis(
        BalancedStrandAdapterVersions.LA_v5and6,
        trimmer_histogram_csv=input_histogram_LAv5and6_csv,
        sorter_stats_csv=sorter_stats_LAv5and6_csv,
        output_path=tmpdir,
        output_basename="TEST_LAv56",
        collect_statistics_kwargs={"input_material_ng": 10},
    )

    balanced_strand_analysis(
        BalancedStrandAdapterVersions.LA_v5,
        trimmer_histogram_csv=input_histogram_LAv5_csv,
        sorter_stats_csv=sorter_stats_LAv5_csv,
        output_path=tmpdir,
        output_basename="TEST_LAv5",
        collect_statistics_kwargs={"input_material_ng": 10},
    )
