import logging

# create logger
import sys
from os.path import dirname

logger = logging.getLogger('systematic_error_correction')
logger.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

base_dir = dirname(__file__)
