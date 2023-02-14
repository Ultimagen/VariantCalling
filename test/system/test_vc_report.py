import shutil
import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir


def test_vc_report(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")

    shutil.copy(pjoin(report_path, "createVarReport.ipynb"), tmpdir)
    shutil.copy(pjoin(report_path, "nexusplt.py"), tmpdir)
    shutil.copy(pjoin(datadir, "test.untrained.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "test.trained.h5"), tmpdir)
    with open(pjoin(tmpdir, "var_report.config"), "w") as config_file:
        strg = """
            [VarReport]
            verbosity = 5
            run_id = 140479-BC21
            pipeline_version = 2.4.1
            reference_version = hg38
            h5_concordance_file = ./test.untrained.h5
            h5_model_file = ./test.trained.h5
            const_model = untrained_ignore_gt_excl_hpol_runs
            trained_model = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
            h5_output = test.var_report.h5
            model_name_with_gt = untrained_ignore_gt_excl_hpol_runs
            model_name_without_gt = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
            model_pkl_with_gt = dummy1.pkl
            model_pkl_without_gt = dummy2.pkl

        """
        config_file.write(strg)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "createVarReport.ipynb",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "createVarReport.nbconvert.ipynb",
        "--template",
        "full",
        "--no-input",
        "--output",
        "varReport.html",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0


def test_vc_report_wo_ref(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")

    shutil.copy(pjoin(report_path, "createVarReport.ipynb"), tmpdir)
    shutil.copy(pjoin(report_path, "nexusplt.py"), tmpdir)
    shutil.copy(pjoin(datadir, "test.untrained.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "test.trained.h5"), tmpdir)
    with open(pjoin(tmpdir, "var_report.config"), "w") as config_file:
        strg = """
            [VarReport]
            verbosity = 5
            run_id = 140479-BC21
            pipeline_version = 2.4.1
            h5_concordance_file = ./test.untrained.h5
            h5_model_file = ./test.trained.h5
            const_model = untrained_ignore_gt_excl_hpol_runs
            trained_model = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
            h5_output = test.var_report.h5
            model_name_with_gt = untrained_ignore_gt_excl_hpol_runs
            model_name_without_gt = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
            model_pkl_with_gt = dummy1.pkl
            model_pkl_without_gt = dummy2.pkl

        """
        config_file.write(strg)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "createVarReport.ipynb",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "createVarReport.nbconvert.ipynb",
        "--template",
        "full",
        "--no-input",
        "--output",
        "varReport.html",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0


def test_vc_report_with_sec(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")
    shutil.copy(pjoin(report_path, "createVarReport.ipynb"), tmpdir)
    shutil.copy(pjoin(report_path, "nexusplt.py"), tmpdir)
    shutil.copy(pjoin(datadir, "test.trained.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "test.ug_hcr.sec.comp.h5"), tmpdir)
    with open(pjoin(tmpdir, "var_report.config"), "w") as config_file:
        strg = """
            [VarReport]
            verbosity = 5
            run_id = 140479-BC21
            pipeline_version = 2.4.1
            reference_version = hg38
            h5_concordance_file = ./test.ug_hcr.sec.comp.h5
            h5_model_file = ./test.trained.h5
            const_model = untrained_ignore_gt_excl_hpol_runs
            trained_model = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
            h5_output = test.var_report.h5
            model_name_with_gt = untrained_ignore_gt_excl_hpol_runs
            model_name_without_gt = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
            model_pkl_with_gt = dummy1.pkl
            model_pkl_without_gt = dummy2.pkl

        """
        config_file.write(strg)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "createVarReport.ipynb",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "createVarReport.nbconvert.ipynb",
        "--template",
        "full",
        "--no-input",
        "--output",
        "varReport.html",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
