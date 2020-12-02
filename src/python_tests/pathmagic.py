from os.path import dirname
from os.path import join as pjoin
import sys

path = pjoin(dirname(__file__), "..")
if not path in sys.path:
    sys.path.append(path)


PYTHON_TESTS_PATH = dirname(__file__)
