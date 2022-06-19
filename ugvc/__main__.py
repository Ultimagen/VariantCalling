# noqa: W605  flake8: invalid escape sequence '\ ' - used by logo
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=wrong-import-position

import sys
from os.path import dirname

from simppl import cli

# add package under repo-root to PYTHONPATH
path = f"{dirname(dirname(__file__))}"
if path not in sys.path:
    sys.path.insert(0, path)

from ugvc.pipelines import (
    coverage_analysis,
    evaluate_concordance,
    filter_variants_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
)
from ugvc.pipelines.mrd import (
    collect_coverage_per_motif,
    concat_dataframes,
    create_control_signature,
    featuremap_to_dataframe,
    intersect_featuremap_with_signature,
    positional_error_rate_profile,
    prepare_data_from_mrd_pipeline,
    snp_error_rate,
)

# import pipeline modules implementing run(argv) method
from ugvc.pipelines.sec import assess_sec_concordance, correct_systematic_errors, sec_training, sec_validation

# create a list of imported pipeline modules
modules = [
    assess_sec_concordance,
    coverage_analysis,
    evaluate_concordance,
    filter_variants_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
]

sec_modules = [correct_systematic_errors, sec_training, sec_validation]

mrd_modules = [
    snp_error_rate,
    positional_error_rate_profile,
    collect_coverage_per_motif,
    concat_dataframes,
    create_control_signature,
    featuremap_to_dataframe,
    intersect_featuremap_with_signature,
    prepare_data_from_mrd_pipeline,
]

modules.extend(mrd_modules)
modules.extend(sec_modules)


LOGO = """
      __    __    ___________    ____  ______
    |  |  |  |  /  _____\   \  /   / /      | # noqa: W605
    |  |  |  | |  |  __  \   \/   / |  ,----' # noqa: W605
    |  |  |  | |  | |_ |  \      /  |  |      # noqa: W605
    |  `--'  | |  |__| |   \    /   |  `----. # noqa: W605
     \______/   \______|    \__/     \______| # noqa: W605

    Ultima genomics variant calling toolkit
    """

# initialize cli
ugvc_cli = cli.CommandLineInterface(__file__, LOGO, modules)
ugvc_cli.run(sys.argv)
