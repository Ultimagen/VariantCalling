import filecmp
import os
import subprocess
from os.path import join as pjoin
from test import get_resource_dir

from ugvc import base_dir

resources_dir = get_resource_dir(__file__)
script_path = pjoin(base_dir, "cnv/cnv_calling_using_cnmops.R")


def test_cnv_calling_using_cnmops(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "merged_cohort_reads_count.rds")
    expected_out_merged_reads_count_file = pjoin(resources_dir, "expected_cohort.cnmops.cnvs.csv")

    out_file = pjoin(tmpdir, "cohort.cnmops.cnvs.csv")
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
        "-minWidth",
        "2",
        "-p",
        "1",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert filecmp.cmp(out_file, expected_out_merged_reads_count_file)
