import logging

# create logger
import sys
from os.path import dirname

logger = logging.getLogger("ugvc")
logger.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stderr)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

base_dir = dirname(__file__)

paths = [f"{dirname(dirname(__file__))}"]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
