# VariantCalling
Variant calling with Ultima data

## Setup
1. Clone VariantCalling repository to e.g. `software\VariantCalling`

2. Create the three conda environments:
  ```
  conda env create -f setup/environment.yml
  conda env create -f setup/other_envs/ucsd.yml
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

See individual tool's documentation pages

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
