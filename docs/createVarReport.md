# Creating a Variant Calling Report

## Input requirements

- run_id: run identifier
- h5_concordance_file: hdf5 file created as output of `run_comparison_pipeline.py`
- h5_model_file: hdf5 file created as output of `train_models_pipeline.py`
- const_model = default model, use `untrained_ignore_gt_excl_hpol_runs`
- trained_model = trained model, use `threshold_model_recall_precision_ignore_gt_excl_hpol_runs`

Input parameters should be contained in a a config file named `var_report.config`, under a `[VarReport]` block, e.g.:

```
[VarReport]
run_id = 140185

h5_concordance_file = 140185.chr9_bam.h5
h5_model_file = 140185.chr9_bam.model.h5

const_model = untrained_ignore_gt_excl_hpol_runs
trained_model = threshold_model_recall_precision_ignore_gt_excl_hpol_runs
```

## Report creation

Execute the Jupyter notebook using the parameters defined in the config file (in the local directory), and convert to html:

```
jupyter nbconvert --to notebook --execute createVarReport.ipynb
jupyter nbconvert --to html createVarReport.nbconvert.ipynb --template full --no-input --output 140185.var_report.html
```
