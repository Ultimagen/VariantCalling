## evaluate_concordance.py

Receives a result of comparing a callset to the ground truth and outputs
evaluation metrics in `h5` file.

**See also**: [run_comparison_pipeline.py](docs/run_comparison_pipeline.md)

### Usage
```
usage: evaluate_concordance.py [-h] --input_file INPUT_FILE --output_prefix
                               OUTPUT_PREFIX [--dataset_key DATASET_KEY]
                               [--ignore_genotype]
                               [--ignore_filters IGNORE_FILTERS]
                               [--output_bed]
                               [--use_for_group_testing USE_FOR_GROUP_TESTING]

Calculate precision and recall for compared HDF5

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Name of the input h5 file (result of run_comparison_pipeline.py)
  --output_prefix OUTPUT_PREFIX
                        Prefix to output files
  --dataset_key DATASET_KEY
                        h5 dataset name, such as chromosome name
  --ignore_genotype     ignore genotype when comparing to ground-truth
  --ignore_filters IGNORE_FILTERS
                        comma separated list of filters to ignore
  --output_bed          output bed files of fp/fn/tp per variant-type
  --use_for_group_testing USE_FOR_GROUP_TESTING
                        Column in the h5 to use for variant grouping (or generate
                        default groupings)
```

### Example
```
evaluate_concordance.py  --input_file 004797-UGAv3-51_2.comp.h5 \
                         --dataset_key chr20  \
                         --output_prefix 004797-UGAv3-51_2.concordance
```

The results can be examined in Python:

`pd.read_hdf("004797-UGAv3-51_2.comp.h5",key="optimal_recall_precision")`



|     | tp  | fp  | fn | precision | recall | f1 |
| --- | --- | --- | -- | --------- | ------ | ------ |
| SNP | 747 |  3  |  6 |   0.996   | 0.992  | 0.99401 |
| Non-hmer_INDEL | 36 | 3 | 3 | 0.92308 | 0.92308 | 0.92308 |
| HMER_indel_<=_4 | 14 | 1 | 1 | 0.93333 | 0.93333 | 0.93333 |
| HMER_indel_(4:8) | 5 | 0 | 0 | 1 | 1 | 1 |
| HMER_indel_[8:10] | 9 | 0 | 0 | 1 | 1 | 1 |
| HMER_indel_11:12 | 7 | 0 | 3 | 1 | 0.7 | 0.82353 |
| HMER_indel_>_12 | 0 | 2 | 13 | 0 | 0 | 0 |
| INDELS | 71 | 6 | 20 | 0.92208 | 0.78022 | 0.84524 |
