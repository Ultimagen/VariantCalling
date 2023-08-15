import filecmp
import os
import subprocess
import pandas as pd
import numpy as np
from os.path import join as pjoin
from test import get_resource_dir

from ugvc import base_dir

resources_dir = get_resource_dir(__file__)
script_path = pjoin(base_dir, "cnv/normalize_reads_count.R")

def test_normalize_reads_count(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.rds")
    expected_out_norm_rc = pjoin(resources_dir, "test_rc.norm.cohort_reads_count.norm.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = [
        "conda",
        "run",
        "-n",
        "cn.mops",
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_reads_count_file",
        in_cohort_reads_count_file,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])


def test_normalize_reads_count_with_ploidy(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.rds")
    ploidy_file = pjoin(resources_dir, "test_rc.ploidy")
    expected_out_norm_rc = pjoin(resources_dir, "test_rc.norm.cohort_reads_count.norm.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = [
        "conda",
        "run",
        "-n",
        "cn.mops",
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_reads_count_file",
        in_cohort_reads_count_file,
        "-ploidy",
        ploidy_file,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])

def test_normalize_reads_count_without_chrX(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.noX.rds")
    expected_out_norm_rc = pjoin(resources_dir, "cohort_reads_count_noX.norm.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = [
        "conda",
        "run",
        "-n",
        "cn.mops",
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_reads_count_file",
        in_cohort_reads_count_file,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])

def test_normalize_reads_count_without_chrXchrY(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.noXnoY.rds")
    expected_out_norm_rc = pjoin(resources_dir, "cohort_reads_count.norm.noXnoY.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = [
        "conda",
        "run",
        "-n",
        "cn.mops",
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_reads_count_file",
        in_cohort_reads_count_file,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])
    