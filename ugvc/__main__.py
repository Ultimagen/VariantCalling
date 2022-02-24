from simppl import cli
import sys
from ugvc.pipelines.sec import sec_training
from ugvc.pipelines.sec import correct_systematic_errors
from ugvc.pipelines.sec import assess_sec_concordance
from python.pipelines import evaluate_concordance

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
