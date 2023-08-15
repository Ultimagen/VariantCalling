import filecmp
import os
import subprocess
import pandas as pd
import numpy as np
from os.path import join as pjoin
from test import get_resource_dir

from ugvc import base_dir

resources_dir = get_resource_dir(__file__)
script_path = pjoin(base_dir, "cnv/merge_reads_count_sample_to_cohort.R")


def test_merge_reads_count_sample_to_cohort(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "cohort_gr_obj.rds")
    in_sample_reads_count_file = pjoin(resources_dir, "sample_gr_obj.rds")
    expected_out_merged_reads_count_file = pjoin(resources_dir, "expected_merged_cohort_reads_count.csv")

    out_file = pjoin(tmpdir, "merged_cohort_reads_count.csv")
    os.chdir(tmpdir)
    cmd = [
        "conda",
        "run",
        "-n",
        "cn.mops",
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_rc",
        in_cohort_reads_count_file,
        "-sample_rc",
        in_sample_reads_count_file,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_merged_reads_count_file)
    assert np.allclose(df.iloc[:, -4], df_ref.iloc[:, -4])
