from os.path import dirname, join as pjoin
import sys

path = dirname(dirname(__file__))
if path not in sys.path:
    sys.path.append(path)


PYTHON_TESTS_PATH = pjoin(dirname(__file__), "data")
COMMON = 'common'
