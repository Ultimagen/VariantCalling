from os.path import dirname

from simppl import cli
import sys

# add package paths to PYTHONPATH
paths = [f'{dirname(dirname(__file__))}/src', f'{dirname(dirname(__file__))}']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

# import pipeline modules implementing run(argv) method
from ugvc.pipelines.sec import sec_training
from ugvc.pipelines.sec import correct_systematic_errors
from ugvc.pipelines.sec import assess_sec_concordance
from python.pipelines import evaluate_concordance

# create a list of imported pipeline modules
modules = [
    sec_training,
    correct_systematic_errors,
    assess_sec_concordance,
    evaluate_concordance
]

logo = \
    """
      __    __    ___________    ____  ______ 
    |  |  |  |  /  _____\   \  /   / /      |
    |  |  |  | |  |  __  \   \/   / |  ,----'
    |  |  |  | |  | |_ |  \      /  |  |     
    |  `--'  | |  |__| |   \    /   |  `----.
     \______/   \______|    \__/     \______|
                                            
    Ultima genomics variant calling toolkit
    """

# initialize cli
ugvc_cli = cli.CommandLineInterface(__file__, logo, modules)
ugvc_cli.run(sys.argv)
