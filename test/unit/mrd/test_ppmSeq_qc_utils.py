from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ugvc.mrd.ppmSeq_utils import (
    add_strand_ratios_and_categories_to_featuremap,
    collect_statistics,
    plot_ppmSeq_strand_ratio,
    plot_strand_ratio_category,
    plot_strand_ratio_category_concordnace,
    plot_trimmer_histogram,
    ppmSeq_qc_analysis,
    ppmSeqAdapterVersions,
    read_ppmSeq_trimmer_histogram,
)

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
input_histogram_legacy_v5_csv = pjoin(
    inputs_dir,
    "130713_UGAv3-51.trimming.A_hmer_5.T_hmer_5.A_hmer_3.T_hmer_3.native_adapter_with_leading_C.histogram.csv",
)
parsed_histogram_legacy_v5_parquet = pjoin(inputs_dir, "130713_UGAv3-51.parsed_histogram.parquet")
sorter_stats_legacy_v5_csv = pjoin(inputs_dir, "130713-UGAv3-51.sorter_stats.csv")
collected_stats_legacy_v5_h5 = pjoin(inputs_dir, "130713-UGAv3-51.stats.h5")
input_histogram_legacy_v5_start_csv = pjoin(inputs_dir, "130715_UGAv3-132.trimming.A_hmer.T_hmer.histogram.csv")
parsed_histogram_legacy_v5_start_parquet = pjoin(inputs_dir, "130715_UGAv3-132.parsed_histogram.parquet")
sorter_stats_legacy_v5_start_csv = pjoin(inputs_dir, "130715-UGAv3-51.sorter_stats.csv")
collected_stats_legacy_v5_start_h5 = pjoin(inputs_dir, "130715-UGAv3-51.stats.h5")
input_featuremap_legacy_v5 = pjoin(inputs_dir, "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.vcf.gz")
expected_output_featuremap_legacy_v5 = pjoin(
    inputs_dir,
    "333_CRCs_39_legacy_v5.featuremap.single_substitutions.subsample.with_strand_ratios.vcf.gz",
)
sorter_stats_csv_ppmSeq_v2 = pjoin(inputs_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.csv")
sorter_stats_json_ppmSeq_v2 = pjoin(inputs_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.json")
trimmer_failure_codes_csv_ppmSeq_v2 = pjoin(
    inputs_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.failure_codes.csv"
)
trimmer_histogram_ppmSeq_v2_csv = pjoin(
    inputs_dir,
    "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT."
    "Start_loop.Start_loop.End_loop.End_loop.native_adapter.histogram.csv",
)
parsed_histogram_parquet_ppmSeq_v2 = pjoin(
    inputs_dir,
    "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.parsed_histogram.parquet",
)

sorter_stats_csv_ppmSeq_v2_amp = pjoin(inputs_dir, "400808-Lb_2768-Z0035-CTGAATGATCTCGAT.csv")
sorter_stats_json_ppmSeq_v2_amp = pjoin(inputs_dir, "400808-Lb_2768-Z0035-CTGAATGATCTCGAT.json")
trimmer_failure_codes_csv_ppmSeq_v2_amp = pjoin(inputs_dir, "400808-Lb_2768-Z0035-CTGAATGATCTCGAT.failure_codes.csv")
trimmer_histogram_ppmSeq_v2_amp = pjoin(
    inputs_dir,
    "400808-Lb_2768-Z0035-CTGAATGATCTCGAT."
    "Start_loop_name.Start_loop_pattern_fw.End_loop_name.End_loop_pattern_fw.Stem_end_length.histogram.csv",
)
trimmer_histogram_extra_ppmSeq_v2_amp = pjoin(
    inputs_dir,
    "400762-Lb_2752-Z0123-CAGATCGCCACAGAT.subsample.Dumbbell_leftover_start_match.hist.csv",
)  # it's not the same file but the right format

subdir = pjoin(inputs_dir, "401057001")
sorter_stats_csv_ppmSeq_v2_401057001 = pjoin(subdir, "401057001-Lb_2772-Z0016-CATCCTGTGCGCATGAT.csv")
sorter_stats_json_ppmSeq_v2_401057001 = pjoin(subdir, "401057001-Lb_2772-Z0016-CATCCTGTGCGCATGAT.json")
trimmer_failure_codes_csv_ppmSeq_v2_401057001 = pjoin(
    subdir, "401057001-Lb_2772-Z0016-CATCCTGTGCGCATGAT_trimmer-failure_codes.csv"
)
trimmer_histogram_ppmSeq_v2_401057001 = pjoin(
    subdir,
    "Z0016-Start_loop_name.Start_loop_pattern_fw.End_loop_name.End_loop_pattern_fw.native_adapter_length.histogram.csv",
)


def _assert_files_are_identical(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        assert f1.read() == f2.read()


def test_read_ppmSeq_v2_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    df_trimmer_histogram = read_ppmSeq_trimmer_histogram(
        ppmSeqAdapterVersions.V1,
        trimmer_histogram_ppmSeq_v2_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
    df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_parquet_ppmSeq_v2)
    assert_frame_equal(
        df_trimmer_histogram,
        df_trimmer_histogram_expected,
    )
    assert_frame_equal(
        df_trimmer_histogram_from_parquet,
        df_trimmer_histogram_expected,
    )


def test_read_ppmSeq_v1_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")
    df_trimmer_histogram = read_ppmSeq_trimmer_histogram(
        ppmSeqAdapterVersions.LEGACY_V5,
        input_histogram_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
    df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    assert_frame_equal(
        df_trimmer_histogram,
        df_trimmer_histogram_expected,
    )
    assert_frame_equal(
        df_trimmer_histogram_from_parquet,
        df_trimmer_histogram_expected,
    )


def test_read_ppmSeq_legacy_v5_trimmer_histogram(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.parquet")

    df_trimmer_histogram = read_ppmSeq_trimmer_histogram(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        input_histogram_legacy_v5_start_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
    df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
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
    df_trimmer_histogram = read_ppmSeq_trimmer_histogram(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        input_histogram_legacy_v5_start_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    plot_trimmer_histogram(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        df_trimmer_histogram,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    df_trimmer_histogram = read_ppmSeq_trimmer_histogram(
        ppmSeqAdapterVersions.LEGACY_V5,
        input_histogram_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    plot_trimmer_histogram(
        ppmSeqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )


def test_collect_statistics(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.h5")
    collect_statistics(
        ppmSeqAdapterVersions.LEGACY_V5,
        trimmer_histogram_csv=input_histogram_legacy_v5_csv,
        sorter_stats_csv=sorter_stats_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )

    f1 = collected_stats_legacy_v5_h5
    f2 = tmp_out_path
    with pd.HDFStore(f1) as fh1, pd.HDFStore(f2) as fh2:
        assert sorted(fh1.keys()) == sorted(fh2.keys())
        keys = fh1.keys()
    for k in keys:
        assert_frame_equal(
            pd.read_hdf(f1, k).rename(
                columns={"native_adapter_with_leading_C": "native_adapter"}
            ),  # backwards compatibility patch for old test data
            pd.read_hdf(f2, k),
        )

    collect_statistics(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        trimmer_histogram_csv=input_histogram_legacy_v5_start_csv,
        sorter_stats_csv=sorter_stats_legacy_v5_csv,
        output_filename=tmp_out_path,
        legacy_histogram_column_names=True,
    )
    f1 = collected_stats_legacy_v5_start_h5
    f2 = tmp_out_path
    with pd.HDFStore(f1) as fh1, pd.HDFStore(f2) as fh2:
        assert fh1.keys() == fh2.keys()
        keys = fh1.keys()
    for k in keys:
        assert_frame_equal(
            pd.read_hdf(f1, k),
            pd.read_hdf(f2, k),
        )


def test_plot_ppmSeq_ratio(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_legacy_v5_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    plot_ppmSeq_strand_ratio(
        ppmSeqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram_legacy_v5_expected,
        output_filename=tmp_out_path,
        title="test",
    )
    df_trimmer_histogram_legacy_v5_start_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    plot_ppmSeq_strand_ratio(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        df_trimmer_histogram_legacy_v5_start_expected,
        output_filename=tmp_out_path,
        title="test",
    )


def test_plot_strand_ratio_category(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_legacy_v5_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    plot_strand_ratio_category(
        ppmSeqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram_legacy_v5_expected,
        title="test",
        output_filename=tmp_out_path,
        ax=None,
    )
    df_trimmer_histogram_legacy_v5_start_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    plot_strand_ratio_category(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        df_trimmer_histogram_legacy_v5_start_expected,
        title="test",
        output_filename=tmp_out_path,
        ax=None,
    )


def test_add_strand_ratios_and_categories_to_featuremap(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.vcf.gz")
    add_strand_ratios_and_categories_to_featuremap(
        ppmSeqAdapterVersions.LEGACY_V5,
        input_featuremap_vcf=input_featuremap_legacy_v5,
        output_featuremap_vcf=tmp_out_path,
    )
    _assert_files_are_identical(
        expected_output_featuremap_legacy_v5,
        tmp_out_path,
    )

    # TODO: instead of identical files:
    # load tmp_out_path
    # check we have strand_ratio and categories in featuremap info
    # check that the values makes sense - perhaps sum on values


def test_plot_strand_ratio_category_concordnace(tmpdir):
    tmp_out_path = pjoin(tmpdir, "tmp_out.png")
    df_trimmer_histogram_legacy_v5_expected = pd.read_parquet(parsed_histogram_legacy_v5_parquet)
    plot_strand_ratio_category_concordnace(
        ppmSeqAdapterVersions.LEGACY_V5,
        df_trimmer_histogram_legacy_v5_expected,
        title="test",
        output_filename=tmp_out_path,
        axs=None,
    )
    df_trimmer_histogram_legacy_v5_start_expected = pd.read_parquet(parsed_histogram_legacy_v5_start_parquet)
    with pytest.raises(ValueError):
        plot_strand_ratio_category_concordnace(
            ppmSeqAdapterVersions.LEGACY_V5_START,
            df_trimmer_histogram_legacy_v5_start_expected,
            title="test",
            output_filename=tmp_out_path,
            axs=None,
        )


def test_ppmSeq_analysis_legacy_v5(tmpdir):
    ppmSeq_qc_analysis(
        ppmSeqAdapterVersions.LEGACY_V5,
        trimmer_histogram_csv=[input_histogram_legacy_v5_csv],
        sorter_stats_csv=sorter_stats_legacy_v5_csv,
        output_path=tmpdir,
        output_basename="TEST_legacy_v5",
        collect_statistics_kwargs={},
        legacy_histogram_column_names=True,
    )

    ppmSeq_qc_analysis(
        ppmSeqAdapterVersions.LEGACY_V5_START,
        trimmer_histogram_csv=[input_histogram_legacy_v5_start_csv],
        sorter_stats_csv=sorter_stats_legacy_v5_start_csv,
        output_path=tmpdir,
        output_basename="TEST_legacy_v5_start",
        collect_statistics_kwargs={},
        legacy_histogram_column_names=True,
    )


def test_ppmSeq_analysis_dmbl(tmpdir):
    ppmSeq_qc_analysis(
        ppmSeqAdapterVersions.DMBL,
        trimmer_histogram_csv=[trimmer_histogram_ppmSeq_v2_amp],
        trimmer_histogram_extra_csv=[trimmer_histogram_extra_ppmSeq_v2_amp],
        sorter_stats_csv=sorter_stats_csv_ppmSeq_v2_amp,
        trimmer_failure_codes_csv=trimmer_failure_codes_csv_ppmSeq_v2_amp,
        output_path=tmpdir,
        output_basename="TEST_DMBL",
    )


def test_ppmSeq_analysis_v1(tmpdir):
    ppmSeq_qc_analysis(
        ppmSeqAdapterVersions.V1,
        trimmer_histogram_csv=[trimmer_histogram_ppmSeq_v2_401057001],
        sorter_stats_csv=sorter_stats_csv_ppmSeq_v2_401057001,
        trimmer_failure_codes_csv=trimmer_failure_codes_csv_ppmSeq_v2_401057001,
        output_path=tmpdir,
        output_basename="TEST_v1",
    )
