import shutil
import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir


def test_ssqc_create_html_report(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")

    # copy notebook
    shutil.copy(pjoin(report_path, "single_sample_qc_create_html_report.ipynb"), tmpdir)
    # copy table CSV
    shutil.copy(pjoin(report_path, "top_metrics_for_tbl.csv"), tmpdir)
    # copy input
    shutil.copy(pjoin(datadir, "input_for_html_report.h5"), tmpdir)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "single_sample_qc_create_html_report.ipynb",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "single_sample_qc_create_html_report.nbconvert.ipynb",
        "--template",
        "classic",
        "--no-input",
        "--output",
        "single_sample_qc_create_html_report.html",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
