# VariantCalling
This package provides a set of tools to assist variant calling on Ultima data. 
The best practice pipeline is published [here](link currently underfined). The code
below is used mostly in the post-GATK filtering step. 

In addition, the code provides 

* python tools to perform evaluation of the callset relative to the ground truth.
* python tools to perform building a database of noisy locaitons (SEC) and filtering callset relative to them.
* Set of tools for MRD (minimal residual disease) - still undocumented.

## Setup
1. Clone VariantCalling repository to e.g. `software\VariantCalling`

2. Create the three conda environments:
  ```
  conda env create -f setup/environment.yml
  conda env create -f setup/other_envs/ucsc.yml
  conda env create -f setup/other_envs/cutadapt.yml
  ```
3. Activate the main conda environment

  ```
  conda activate genomics.py3
  ```

4. Install `ugvc` package

   a. User

      ```
      cd VariantCalling
      pip install .
      ```

   b. Developer
      Install in editable mode
      ```
      cd VariantCalling
      pip install -e .
      ```

 scripts should be available on the path and modules should be available for import through from ugvc import ...

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

	assess_sec_concordance:
		Given a concordance h5 input, an exclusion candidates bed, and a SEC refined exclude-list (bed)
		Apply each exclusion list on the variants and measure the differences between the results.

	correct_systematic_errors:
		filter out variants which appear like systematic-errors, while keeping those which are not well explained by errors

	sec_training:
		SEC (Systematic Error Correction) training pipeline

	sec_validation:
		SEC (Systematic Error Correction) validation pipeline

## Documentation of individual tools: 

* [coverage_analysis](docs/coverage_analysis.md)
* [evaluate_concordance](docs/evaluate_concordance.md)
* [filter_variants_pipeline](docs/filter_variants_pipeline.md)
* [run_comparison_pipeline](docs/run_comparison_pipeline.md)
* [train_models_pipeline](docs/train_models_pipeline.md)

## Test
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
