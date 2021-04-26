import pathmagic
from pathmagic import PYTHON_TESTS_PATH
from os.path import join as pjoin
import shutil
import subprocess


def test_vc_report(tmpdir):
    path = pathmagic.path
    datadir = pjoin(pathmagic.PYTHON_TESTS_PATH, "reports")
    report_path = pjoin(path, "python", "reports")
    shutil.copy(pjoin(report_path, "createVarReport.ipynb"), tmpdir)
    shutil.copy(pjoin(report_path, "nexusplt.py"), tmpdir)
    shutil.copy(pjoin(datadir, "test.untrained.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "test.trained.h5"), tmpdir)
    with open(pjoin(tmpdir, "var_report.config"), 'w') as config_file:
        strg = '''
        [VarReport]
        run_id = 140479-BC21
        pipeline_version = 2.4.1
        h5_concordance_file = ./test.untrained.h5
        h5_model_file = ./test.trained.h5

        const_model = untrained_ignore_gt_excl_hpol_runs

        trained_model = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
        h5_output = test.var_report.h5
        '''
        config_file.write(strg)

    cmd = ['jupyter', 'nbconvert', '--to', 'notebook',
           '--execute', 'createVarReport.ipynb']
    assert (subprocess.check_call(cmd, cwd=tmpdir) == 0)

    cmd = ['jupyter', 'nbconvert', '--to', 'html', 'createVarReport.nbconvert.ipynb', '--template', 'full',
           '--no-input', '--output', 'varReport.html']
    assert (subprocess.check_call(cmd, cwd=tmpdir) == 0)
