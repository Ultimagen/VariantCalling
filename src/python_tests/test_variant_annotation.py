import pathmagic
import pytest
import subprocess
import pysam
import pyfaidx
import numpy as np
import pandas as pd
from os.path import join as pjoin
import python.modules.variant_annotation as variant_annotation
from pathmagic import PYTHON_TESTS_PATH


def test_get_coverage(tmpdir):
    temp_bam_name1 = _create_temp_bam(tmpdir, "test1.bam")
    temp_bam_name2 = _create_temp_bam(tmpdir, "test2.bam")
    df = _create_test_df()

    result = variant_annotation.get_coverage(
        df.copy(), [temp_bam_name1, temp_bam_name2], 20, True)
    print(result.head())
    expected_total, expected_well_mapped = _create_expected_coverage()
    assert result.shape == (df.shape[0], df.shape[1] + 3)
    assert 'coverage' in result.columns
    assert 'well_mapped_coverage' in result.columns
    assert 'repetitive_read_coverage' in result.columns
    assert np.all(result['coverage'] - result['well_mapped_coverage'] == result['repetitive_read_coverage'])
    assert np.all(result['coverage'] == 2*expected_total)
    assert np.all(result['well_mapped_coverage'] == 2*expected_well_mapped)


def _create_temp_bam(tmpdir, name):
    header = {'HD': {'VN': '1.0'},
              'SQ': [{'LN': 100000, 'SN': 'chr20'}]}
    fai = pyfaidx.Fasta(pjoin(PYTHON_TESTS_PATH, "common", "sample.fasta"))
    with pysam.AlignmentFile(pjoin(tmpdir, name), "wb", header=header) as outf:
        for i in range(90000, 91000):
            a = pysam.AlignedSegment()
            a.query_name = f"read_{i}"
            a.query_sequence = str(fai['chr20'][i])
            a.query_qualities = pysam.qualitystring_to_array("I")
            a.is_duplicate = i % 2
            a.reference_id = 0
            a.reference_start = i
            a.cigar = ((0, 1),)
            a.mapping_quality = 30 * (i % 3)
            outf.write(a)
    subprocess.check_call(['samtools', 'index', pjoin(tmpdir, name)])
    return pjoin(tmpdir, name)


def _create_test_df():
    df = pd.DataFrame(
        {'chrom': ['chr20'] * 2000, 'pos': np.arange(90000, 92000)+1})  # note  that VCF is one-based
    return df


def _create_expected_coverage():
    result = np.ones(1000, dtype=np.int)
    result[((np.arange(90000, 91000)) % 2) == 1] = 0
    result_well_mapped = result.copy()
    result_well_mapped[((np.arange(90000, 91000)) % 3) == 0] = 0
    result = np.concatenate((result, np.zeros(1000)))
    result_well_mapped = np.concatenate((result_well_mapped, np.zeros(1000)))
    return result, result_well_mapped
