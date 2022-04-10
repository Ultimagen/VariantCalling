from os.path import exists
from os.path import join as pjoin

from ugvc.utils.misc_utils import find_scripts_path


class TestMiscUtils:
    def test_find_scripts_path(self):
        assert exists(pjoin(find_scripts_path(), "run_ucsc_command.sh"))
