import sys
from os.path import dirname

from simppl import cli

# add package paths to PYTHONPATH
paths = [f"{dirname(dirname(__file__))}"]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

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
from ugvc.pipelines.sec import (
    assess_sec_concordance,
    correct_systematic_errors,
    sec_training,
)

# create a list of imported pipeline modules
modules = [
    assess_sec_concordance,
    coverage_analysis,
    evaluate_concordance,
    filter_variants_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
]

sec_modules = [sec_training, correct_systematic_errors]

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

logo = """
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
