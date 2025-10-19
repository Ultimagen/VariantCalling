from os.path import exists
from os.path import join as pjoin

from ugvc.utils.exec_utils import find_scripts_path


def test_find_scripts_path():
    assert exists(pjoin(find_scripts_path(), "run_ucsc_command.sh"))
