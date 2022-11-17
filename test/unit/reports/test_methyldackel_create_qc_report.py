import shutil
import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir


def test_methyldackel_create_qc_report(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")

    # copy notebook ugvc/reports/methyldackel_qc_report.ipynb
    shutil.copy(pjoin(report_path, "methyldackel_qc_report.ipynb"), tmpdir)

    # copy input: input_for_html_report.csv
    shutil.copy(pjoin(datadir, "input_for_html_report.csv"), tmpdir)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "methyldackel_qc_report.ipynb",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "methyldackel_qc_report.nbconvert.ipynb",
        "--template",
        "classic",
        "--no-input",
        "--output",
        "methyldackel_qc_report.html",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0