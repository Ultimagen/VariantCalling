import pandas as pd
import numpy as np
from os.path import join as pjoin, isfile
from tempfile import TemporaryDirectory
from collections.abc import Iterable
import pathmagic
from pathmagic import PYTHON_TESTS_PATH
from python.pipelines.coverage_analysis import (
    calculate_and_bin_coverage,
    create_coverage_annotations,
    COVERAGE
)


f_in = [
    pjoin(
        PYTHON_TESTS_PATH,
        "170201-BC23.chr9_1000000_2001000.aligned.unsorted.duplicates_marked.bam",
    ),
    pjoin(
        PYTHON_TESTS_PATH,
        "170201-BC10.chr9_1000000_2001000.aligned.unsorted.duplicates_marked.bam",
    ),
]
f_annotations_sum = pjoin(PYTHON_TESTS_PATH, "coverage_annotations_sum.h5")
f_in_gs = "gs://runs-data/cromwell-execution/JukeboxVC/523a5c47-d720-49ea-8e2f-0d2a779bd63a/call-MarkDuplicatesSpark/cacheCopy/170201-BC23.aligned.unsorted.duplicates_marked.bam"
regions = ["chr9:1000000-1001000", "chr9:2000000-2001000"]
region_large = "chr9:1000000-2000000"
window = 100
mapq_th = 60
bq_th = 20
rl_th = 200
url_th = 200
expected_outputs = [
    329.41,
    262.39,
    365.04,
    233.54,
    261.64,
    318.94,
    32.91,
    296.5,
    333469.88,
    329.41 + 365.04,
]


def _get_fout(tmpdir):
    return [
        pjoin(tmpdir, f"170201-BC23.chr9_1000000-1001000.w{window}.parquet"),
        pjoin(tmpdir, f"170201-BC10.chr9_1000000-1001000.w{window}.parquet"),
        pjoin(tmpdir, f"170201-BC23.chr9_2000000-2001000.w{window}.parquet"),
        pjoin(tmpdir, f"170201-BC10.chr9_2000000-2001000.w{window}.parquet"),
        pjoin(tmpdir, f"170201-BC10.chr9_1000000-1001000.w{window}.Q{mapq_th}.parquet"),
        pjoin(tmpdir, f"170201-BC23.chr9_1000000-1001000.w{window}.q{bq_th}.parquet"),
        pjoin(tmpdir, f"170201-BC23.chr9_1000000-1001000.w{window}.l{rl_th}.parquet"),
        pjoin(tmpdir, f"170201-BC23.chr9_1000000-1001000.w{window}.L{url_th}.parquet"),
        pjoin(tmpdir, f"170201-BC23.chr9_1000000-2000000.w{window}.parquet"),
        pjoin(tmpdir, f"170201-BC23.merged_regions.w{window}.parquet"),
    ]


def _assert_fout(f_out_, j):
    if isinstance(j, Iterable):
        [_assert_fout(f_out_, j_) for j_ in j]
    else:
        assert isfile(f_out_[j])
        assert np.allclose(
            pd.read_parquet(f_out_[j]).sum()[COVERAGE], expected_outputs[j]
        )


def test_coverage_analysis():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0], f_out=tmpdir, region=regions[0], window=window, n_jobs=1
        )
        j = 0
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_coverage_analysis_multiple_files():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in, f_out=tmpdir, region=regions[0], window=window, n_jobs=1
        )
        j = [0, 1]
        assert output == [f_out[k] for k in j]
        _assert_fout(f_out, j)


def test_coverage_analysis_multiple_regions():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0], f_out=tmpdir, region=regions, window=window, n_jobs=1
        )
        j = [0, 2]
        assert output == [f_out[k] for k in j]
        _assert_fout(f_out, j)


def test_coverage_analysis_multiple_regions_with_merge():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0],
            f_out=tmpdir,
            region=regions,
            window=window,
            n_jobs=1,
            merge_regions=True,
        )
        j = 9
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_coverage_analysis_multiple_files_and_regions():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in, f_out=tmpdir, region=regions, window=window, n_jobs=1
        )
        j = [0, 1, 2, 3]
        assert output[0][0] == f_out[j[0]]
        assert output[0][1] == f_out[j[2]]
        assert output[1][0] == f_out[j[1]]
        assert output[1][1] == f_out[j[3]]
        _assert_fout(f_out, j)


def test_coverage_analysis_mapq_th():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[1],
            f_out=tmpdir,
            region=regions[0],
            window=window,
            n_jobs=1,
            min_mapq=mapq_th,
        )
        j = 4
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_coverage_analysis_bq_th():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0],
            f_out=tmpdir,
            region=regions[0],
            window=window,
            n_jobs=1,
            min_bq=bq_th,
        )
        j = 5
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_coverage_analysis_rl_th():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0],
            f_out=tmpdir,
            region=regions[0],
            window=window,
            n_jobs=1,
            min_read_length=rl_th,
        )
        j = 6
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_coverage_analysis_max_length():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0],
            f_out=tmpdir,
            region=regions[0],
            window=window,
            max_read_length=url_th,
            n_jobs=1,
        )
        j = 7
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_coverage_analysis_with_gs_access():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in_gs, f_out=tmpdir, region=regions[0], window=window, n_jobs=1
        )
        j = 0
        assert output == f_out[j]
        _assert_fout(f_out, j)


def test_create_coverage_annotations():
    with TemporaryDirectory() as tmpdir:
        f_out = _get_fout(tmpdir)
        output = calculate_and_bin_coverage(
            f_in[0], f_out=tmpdir, region=region_large, window=window, n_jobs=1
        )
        j = 8
        assert output == f_out[j]
        _assert_fout(f_out, j)
        f_out_annot = pjoin(
            tmpdir, f"annotations.170201-BC23.chr9_1000000-2000000.w{window}.parquet"
        )
        create_coverage_annotations(f_out[8])
        assert isfile(f_out_annot)
        df = pd.read_parquet(f_out_annot).set_index(["chrom", "chromStart", "chromEnd"])
        annotations_sum = pd.read_hdf(f_annotations_sum)
        assert np.all(annotations_sum == df.sum())
