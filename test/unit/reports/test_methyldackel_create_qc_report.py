import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir


def test_methyldackel_create_qc_report(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")
    notebook_file_in = pjoin(report_path, "methyldackel_qc_report.ipynb")
    papermill_out = pjoin(tmpdir, "methyldackel_qc_report.papermill.ipynb")
    input_csv_file = pjoin(datadir, "input_for_html_report.csv")
    base_file_name = "test"

    cmd1 = (
        f"papermill {notebook_file_in} {papermill_out} "
        f"-p input_csv_file {input_csv_file} "
        f"-p input_base_file_name {base_file_name}"
    )
    assert subprocess.check_call(cmd1.split(), cwd=tmpdir) == 0

    cmd2 = (
        f"jupyter nbconvert --to html {papermill_out} " f"--template classic --no-input --output {base_file_name}.html"
    )
    assert subprocess.check_call(cmd2.split(), cwd=tmpdir) == 0
