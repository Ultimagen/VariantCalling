# VariantCalling
Variant calling with Ultima data

## Setup
Add ugvc and src/python to PYTHONPATH, so that python will be able to find absolute imports
```
export PYTHONPATH="$PYTHONPATH:path/to/repo:/path/to/repo/src/"
```

Create the three conda environments:
```
conda env create -f setup/environment.yml
conda env create -f setup/other_envs/ucsv.yml
conda env create -f setup/other_envs/cutadapt.yml
```

## Test
```
python -m pytest
```
Notice that test_db_access needs your machine to have access credentials to mongoDB.
To ignore this test, run:
```
python -m pytest --ignore src/python_tests/test_db_access.py
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


