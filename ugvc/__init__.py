import logging

# create logger
import sys
from os.path import dirname

logger = logging.getLogger('ugvc')
logger.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

base_dir = dirname(__file__)

if base_dir not in sys.path:
    sys.path.append(base_dir)

path = f'{dirname(base_dir)}/src'
if path not in sys.path:
    sys.path.append(path)
