import shutil
import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir


def test_ssqc_create_html_report(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")


    cmd = (
        f"papermill {report_path}/single_sample_qc_create_html_report.ipynb single_sample_qc_create_html_report.papermill.ipynb "
        f"-p top_metrics_file {report_path}/top_metrics_for_tbl.csv "
        f"-p input_h5_file {datadir}/input_for_html_report.h5 "
        f"-p input_base_file_name TEST".split()
    )

    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    jupyter_convert_cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "single_sample_qc_create_html_report.papermill.ipynb",
        "--template",
        "classic",
        "--no-input",
        "--output",
        "single_sample_qc_create_html_report.html",
    ]
    assert subprocess.check_call(jupyter_convert_cmd, cwd=tmpdir) == 0
