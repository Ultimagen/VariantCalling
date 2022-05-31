## train_models_pipeline.py

This script trains an ML filtering model for the GATK raw callset. 

There are two modes of action: 

- Real ground truth: if the input is the result of [run_comparision_pipeline.py](run_comparison_pipeline.md). 
In this case the model is trained on the result of comparision with the given labels
- Approximate ground truth: if the input is the call VCF, the ground truth is estimated by assigning 
calls from dbSNP to true positive class and calls from `blacklist` to false positive class

**See also:** [How to post-filter a callset](howto-callset-filter.md)

### Usage 

```
usage: train_models_pipeline.py [-h] [--input_file INPUT_FILE]
                                [--blacklist BLACKLIST] --output_file_prefix
                                OUTPUT_FILE_PREFIX [--mutect]
                                [--evaluate_concordance]
                                [--apply_model APPLY_MODEL]
                                [--evaluate_concordance_contig EVALUATE_CONCORDANCE_CONTIG]
                                [--input_interval INPUT_INTERVAL]
                                [--list_of_contigs_to_read [LIST_OF_CONTIGS_TO_READ [LIST_OF_CONTIGS_TO_READ ...]]]
                                --reference REFERENCE
                                [--runs_intervals RUNS_INTERVALS]
                                [--annotate_intervals ANNOTATE_INTERVALS]
                                [--exome_weight EXOME_WEIGHT]
                                [--flow_order FLOW_ORDER]
                                [--exome_weight_annotation EXOME_WEIGHT_ANNOTATION]
                                [--vcf_type VCF_TYPE] [--ignore_filter_status]
                                [--verbosity VERBOSITY]

Train filtering models on the concordance file

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Name of the input h5/vcf file. h5 is output of
                        comparison
  --blacklist BLACKLIST
                        blacklist file by which we decide variants as FP
  --output_file_prefix OUTPUT_FILE_PREFIX
                        Output .pkl file with models, .h5 file with results
  --mutect
  --evaluate_concordance
                        Should the results of the model be applied to the
                        concordance dataframe
  --apply_model APPLY_MODEL
                        If evaluate_concordance - which model should be
                        applied
  --evaluate_concordance_contig EVALUATE_CONCORDANCE_CONTIG
                        Which contig the evaluation of the model should be
                        done on
  --input_interval INPUT_INTERVAL
                        bed file of intersected intervals from run_comparison
                        pipeline
  --list_of_contigs_to_read [LIST_OF_CONTIGS_TO_READ [LIST_OF_CONTIGS_TO_READ ...]]
                        List of contigs to read from the DF
  --reference REFERENCE
                        Reference genome
  --runs_intervals RUNS_INTERVALS
                        Runs intervals (bed/interval_list)
  --annotate_intervals ANNOTATE_INTERVALS
                        interval files for annotation (multiple possible)
  --exome_weight EXOME_WEIGHT
                        weight of exome variants in comparison to whole genome
                        variant
  --flow_order FLOW_ORDER
                        Sequencing flow order (4 cycle)
  --exome_weight_annotation EXOME_WEIGHT_ANNOTATION
                        annotation name by which we decide the weight of exome
                        variants
  --vcf_type VCF_TYPE   VCF type - "single_sample" or "joint"
  --ignore_filter_status
                        Ignore the `filter` and `tree_score` columns
  --verbosity VERBOSITY
                        Verbosity: ERROR, WARNING, INFO, DEBUG
```

### Example 

```
train_models_pipeline.py \
    --input_file test.annotated.vcf.gz \
    --reference Homo_sapiens_assembly38.fasta \
    --runs_intervals runs.conservative.bed \
    --blacklist cohort_fp.h5 \
    --flow_order TGCA \
    --exome_weight 100 \
    --exome_weight_annotation exome.twist \
    --annotate_intervals LCR-hs38.bed \
    --annotate_intervals exome.twist.bed \
    --annotate_intervals mappability.0.bed \
    --output_file_prefix test.model  
```

The output of the run is a trained model file called `test.model.pkl` that can be used in [filter_variants_pipeline.py](filter_variants_pipeline.md). 