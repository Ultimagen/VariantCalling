import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir

jupyter_convert_cmd = (
    "jupyter nbconvert --to html createVarReport.nbconvert.ipynb "
    "--template full --no-input --output varReport.html".split()
)
data_dir = get_resource_dir(__file__)
path = test_dir
report_path = pjoin(path, "..", "ugvc", "reports")


def test_vc_report(tmpdir):
    cmd = (
        f"papermill {report_path}/createVarReport.ipynb createVarReport.nbconvert.ipynb "
        "-p verbosity 5 "
        "-p run_id 140479-BC21 "
        "-p pipeline_version 2.4.1 "
        f"-p h5_concordance_file {data_dir}/test.untrained.h5 "
        "-p h5_output h5_output_file "
        "-p model_name untrained_ignore_gt_excl_hpol_runs ".split()
    )
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert subprocess.check_call(jupyter_convert_cmd, cwd=tmpdir) == 0


def test_vc_report_verbosity_1(tmpdir):
    cmd = (
        f"papermill {report_path}/createVarReport.ipynb createVarReport.nbconvert.ipynb "
        "-p verbosity 5 "
        "-p run_id 140479-BC21 "
        "-p pipeline_version 2.4.1 "
        f"-p h5_concordance_file {data_dir}/test.untrained.h5 "
        "-p h5_output h5_output_file "
        "-p model_name untrained_ignore_gt_excl_hpol_runs ".split()
    )
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert subprocess.check_call(jupyter_convert_cmd, cwd=tmpdir) == 0


def test_vc_report_wo_ref(tmpdir):
    cmd = (
        f"papermill {report_path}/createVarReport.ipynb createVarReport.nbconvert.ipynb "
        "-p verbosity 5 "
        "-p run_id 140479-BC21 "
        "-p pipeline_version 2.4.1 "
        f"-p h5_concordance_file {data_dir}/test.untrained.h5 "
        "-p h5_output h5_output_file "
        "-p model_name untrained_ignore_gt_excl_hpol_runs ".split()
    )
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert subprocess.check_call(jupyter_convert_cmd, cwd=tmpdir) == 0


def test_vc_report_with_sec(tmpdir):
    cmd = (
        f"papermill {report_path}/createVarReport.ipynb createVarReport.nbconvert.ipynb "
        "-p verbosity 5 "
        "-p run_id 140479-BC21 "
        "-p pipeline_version 2.4.1 "
        f"-p h5_concordance_file {data_dir}/test.ug_hcr.sec.comp.h5 "
        "-p h5_output h5_output_file "
        "-p model_name threshold_model_recall_precision_ignore_gt_excl_hpol_runs ".split()
    )

    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert subprocess.check_call(jupyter_convert_cmd, cwd=tmpdir) == 0
