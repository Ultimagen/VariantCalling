import filecmp
import subprocess
from os.path import join as pjoin
from test import get_resource_dir

import numpy as np
import pandas as pd

from ugvc import base_dir

resources_dir = get_resource_dir(__file__)
script_path = pjoin(base_dir, "cnv/create_reads_count_cohort_matrix.R")


def test_create_reads_count_cohort_matrix(tmpdir):
    in_rds_file = pjoin(resources_dir, "sample_gr_obj.rds")
    rc_files_list = pjoin(tmpdir, "samples_read_count_files_list")
    with open(rc_files_list, "w") as f:
        f.write(in_rds_file + '\n')
        f.write(in_rds_file + '\n')
        f.write(in_rds_file + '\n')

    expected_out_file = pjoin(resources_dir, "merged_cohort_reads_count.csv")
    out_file = pjoin(tmpdir, "merged_cohort_reads_count.csv")

    cmd = [
        "conda",
        "run",
        "-n",
        "cn.mops",
        "Rscript",
        "--vanilla",
        script_path,
        "-samples_read_count_files_list",
        rc_files_list,
        "--save_csv"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_file)
    assert np.allclose(df.iloc[:,-3:], df_ref.iloc[:,-3:])
