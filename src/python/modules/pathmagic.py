from os.path import dirname
from os.path import join as pjoin
import sys

path = dirname(dirname(dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

path = dirname(path)
if path not in sys.path:
    sys.path.append(path)
