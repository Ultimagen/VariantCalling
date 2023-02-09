# VariantCalling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package provides a set of tools to assist variant calling on Ultima data.
The best practice pipeline is published [here](broad.io/ugworkspace). The code
below is used mostly in the post-GATK filtering step.

In addition, the code provides

* Tools to perform evaluation of the callset relative to the ground truth.
* Tools to perform building a database of noisy locaitons (SEC) and filtering callset relative to them - still undocumented
* Set of tools for MRD (minimal residual disease) - still undocumented.

## Setup
1. Make sure git-lfs is installed on your system if you want to clone test resources along with the code (https://git-lfs.github.com/)
1. Clone VariantCalling repository to e.g. `software/VariantCalling`
1. Create the three conda environments:
   * `conda env create -f setup/environment.yml`
   * `conda env create -f setup/other_envs/ucsc.yml`
   * `conda env create -f setup/other_envs/cutadapt.yml`

1. Activate the main conda environment
   * `conda activate genomics.py3`
1. Install the package
   * `cd software/VariantCalling`
   * `pip install .`

## Using ugvc package

### Run through cli

To get a list of available cli tools:
```
python /path/to/ugvc
```

To run a specific tool:

```
python /path/to/ugvc <tool_name> <args>
```

### Run individual tools not through CLI

	coverage_analysis:
		Run full coverage analysis of an aligned bam/cram file

	evaluate_concordance:
		Calculate precision and recall for compared HDF5

	filter_variants_pipeline:
		POST-GATK variant filtering

	run_comparison_pipeline:
		Concordance between VCF and ground truth

	train_models_pipeline:
		Train filtering models


## Documentation of individual tools:

* Train post-calling model: [train_models_pipeline](docs/train_models_pipeline.md)
* Filter callset using pre-trained ML model: [filter_variants_pipeline](docs/filter_variants_pipeline.md)
* Compare callset to ground truth: [run_comparison_pipeline](docs/run_comparison_pipeline.md)
* Coverage bias analyses: [coverage_analysis](docs/coverage_analysis.md)
* Evaluation of compared callsets: [evaluate_concordance](docs/evaluate_concordance.md)

## Howtos

* [How to post-filter a callset](docs/howto-callset-filter.md)
* [Evaluation of UG callsets](docs/howto-evaluate-ug-callset.md)

## Test

### Recommended way to run tests for external users
```
./run_tests.sh
```
This script will validate that test resources were correctly cloned, and only then run tests

### Run all tests
```
python -m pytest
```
Notice that test_db_access needs your machine to have access credentials to mongoDB.
To ignore this test, run:
```
python -m pytest --ignore test/unit/test_db_access.py
```

### Run unit-tests
```
python -m pytest test/unit
```

### Run system-tests
```
python -m pytest test/system
```

## Git-lfs
Whenever commiting a data-file to the repo, check that it's suffix is tracked by git-lfs in .gitattributes
If not, add the new suffix to the .gitattributes file before adding the data-file and commiting it.
Also make sure to commit .gitattributes itself.
```
git-lfs track "*.new_suffix"
```

## Development guidelines
1. Always develop on a branch, not on master
2. Public functions/classes should be tested, using either pytest or unittest syntax
3. commit and push your changes to that branch on the remote repo
4. Open a pull-request through github
   1. Add at least one code reviewer
   2. Wait for CI tests to pass (green V sign)
5. scripts that you want to be available on the path should be added to `setup.py`
6. scripts that you want to be available to `ugvc` should be added to `__main__.py`
7. Code changes should pass all pre-commit hooks

## How To pre-commit
pre-commit hooks are configured within `.pre-commit-config.yaml`

install: https://pre-commit.com/#installation

After pre-commit package is installed, you need to set git hooks scripts:
```
pre-commit install
pre-commit install -t pre-commit
```
After the installation it will run the pre-commit hooks for all files changed as part of the commit.
This should look like this, notice mostly the red "Failed" issues that you must fix, the pre-commit verifies the fix before enables the commit:
```
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...........................................(no files to check)Skipped
check json...........................................(no files to check)Skipped
check for added large files..............................................Passed
pycln....................................................................Passed
isort....................................................................Passed
black....................................................................Passed
flake8...................................................................Passed
pylint...................................................................Passed
[master 9a1a910e] Test pre-commit
 1 file changed, 1 deletion(-)
 ```
For running all pre-commit hooks on all files (used for initial pre-commit run) use: `pre-commit run --all-files`

# The hooks we use are:
[pycln](https://github.com/hadialqattan/pycln) - remove unused import statements

[isort](https://github.com/PyCQA/isort) - Python utility library to sort imports alphabetically, and automatically separated into sections and by type

[black](https://github.com/psf/black) - uncompromising Python code formatter

[flake8](https://gitlab.com/pycqa/flake8) - python coding style guide for PEP8

[pylint](https://github.com/pycqa/pylint) - python static code analysis tool
