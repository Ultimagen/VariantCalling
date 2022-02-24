from os.path import dirname, join as pjoin
import sys

path = dirname(dirname(__file__))
if path not in sys.path:
    sys.path.append(path)
vc_path = path
path = dirname(path)
if path not in sys.path:
    sys.path.append(path)
ugvc_path = path


PYTHON_TESTS_PATH = pjoin(dirname(__file__), "data")
COMMON = 'common'
