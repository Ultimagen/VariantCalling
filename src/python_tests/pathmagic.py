from os.path import dirname
import sys

path = dirname(dirname(__file__))
if path not in sys.path:
    sys.path.append(path)


PYTHON_TESTS_PATH = dirname(__file__)
