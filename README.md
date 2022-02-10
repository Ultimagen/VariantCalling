# VariantCalling
Variant calling with Ultima data

## Setup
Add ugvc and src/python to PYTHONPATH, so that python will be able to find absolute imports
```
export PYTHONPATH="$PYTHONPATH:path/to/repo:/path/to/repo/src/"
```

Create conda environments:
```
conda env create -f setup/environment.yml
```