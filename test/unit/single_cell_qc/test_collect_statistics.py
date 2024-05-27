import shutil
from gzip import BadGzipFile
from pathlib import Path
from test import get_resource_dir

import pandas as pd
import pytest

from ugvc.pipelines.single_cell_qc.collect_statistics import (
    collect_statistics,
    extract_statistics_table,
    get_insert_properties,
    read_star_stats,
)
from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import H5Keys, Inputs, Thresholds

inputs_dir = Path(
    get_resource_dir(__file__)
)  # returns VariantCalling/test/resources/unit/single_cell_qc + the current file name: test_single_cell_qc_pipeline
inputs_dir = (
    inputs_dir.parent
)  # the resources are in the directory: VariantCalling/test/resources/unit/single_cell_qc so need to drop the last part of the path

inputs = Inputs(
    trimmer_stats_csv=str(Path(inputs_dir) / "trimmer_stats.csv"),
    trimmer_histogram_csv=[str(Path(inputs_dir) / "trimmer_histogram.csv")],
    trimmer_failure_codes_csv=str(Path(inputs_dir) / "trimmer_failure_codes.csv"),
    sorter_stats_csv=str(Path(inputs_dir) / "sorter_stats.csv"),
    star_stats=str(Path(inputs_dir) / "star_insert_Log.final.out"),
    star_reads_per_gene=str(Path(inputs_dir) / "star_insert_ReadsPerGene.out.tab"),
    insert=str(Path(inputs_dir) / "insert_subsample.fastq.gz"),
)
thresholds = Thresholds(
    pass_trim_rate=0.9,
    read_length=100,
    fraction_below_read_length=0.1,
    percent_aligned=0.9,
)
sample_name = "test_sample"


def test_collect_statistics(tmpdir):
    tmpdir = Path(tmpdir)
    output_path = collect_statistics(
        input_files=inputs,
        output_path=tmpdir,
        sample_name=sample_name,
    )

    # assert output file exists
    assert output_path.exists()
    # assert output file is not empty
    assert output_path.stat().st_size > 0
    # assert output file contains expected keys
    with pd.HDFStore(output_path, mode="r") as store:
        expected_keys = [
            "/" + key.value for key in H5Keys if key != H5Keys.STATISTICS_SHORTLIST
        ]
        assert set(store.keys()) == set(expected_keys)


def test_read_star_stats():
    star_stats_file = inputs.star_stats
    df = read_star_stats(star_stats_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (32, 4)
    assert df.isna().sum().sum() == 0


def test_read_star_stats_missing_file():
    star_stats_file = "missing_file.txt"
    with pytest.raises(FileNotFoundError):
        df = read_star_stats(star_stats_file)


def test_get_insert_properties():
    insert = inputs.insert
    df_insert_quality, insert_lengths = get_insert_properties(insert)
    assert isinstance(df_insert_quality, pd.DataFrame)
    assert isinstance(insert_lengths, list)
    assert len(insert_lengths) == 9652
    assert df_insert_quality.shape == (11, 219)


def test_get_insert_properties_no_gzip():
    insert = inputs.star_stats
    with pytest.raises(BadGzipFile):
        get_insert_properties(insert)


def test_get_insert_properties_no_fastq():
    insert = str(Path(inputs_dir) / "trimmer_stats.csv.gz")
    with pytest.raises(BadGzipFile):
        get_insert_properties(insert)


def test_extract_statistics_table(tmpdir):
    original_h5_file = Path(inputs_dir) / "single_cell_qc_stats_no_shortlist.h5"
    # copy the file to tmpdir to avoid modifying the original file
    h5_file = tmpdir / original_h5_file.name
    shutil.copyfile(original_h5_file, str(h5_file))

    # assert h5 doesnt contain the STATISTICS_SHORTLIST key
    with pd.HDFStore(h5_file, mode="r") as store:
        assert "/" + H5Keys.STATISTICS_SHORTLIST.value not in store.keys()

    extract_statistics_table(h5_file)

    # assert h5 contains the STATISTICS_SHORTLIST key
    with pd.HDFStore(h5_file, mode="r") as store:
        assert "/" + H5Keys.STATISTICS_SHORTLIST.value in store.keys()
