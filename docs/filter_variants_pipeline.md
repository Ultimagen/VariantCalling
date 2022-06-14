## filter_variants_pipeline.py

This script receves a raw GATK output VCF file, and applies to it a filtering model that had been trained previously.
In addition, the script can label variants by membership in a "common" false positive calls list.

See also: [How to post-filter a callset](howto-callset-filter.md)

```
usage: filter_variants_pipeline.py [-h] --input_file INPUT_FILE --model_file
                                   MODEL_FILE --model_name MODEL_NAME
                                   [--hpol_filter_length_dist HPOL_FILTER_LENGTH_DIST HPOL_FILTER_LENGTH_DIST]
                                   --runs_file RUNS_FILE
                                   [--blacklist BLACKLIST]
                                   [--blacklist_cg_insertions]
                                   --reference_file REFERENCE_FILE
                                   --output_file OUTPUT_FILE [--is_mutect]
                                   [--flow_order FLOW_ORDER]
                                   [--annotate_intervals ANNOTATE_INTERVALS]

Filter VCF

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Name of the input VCF file
  --model_file MODEL_FILE
                        Pickle model file
  --model_name MODEL_NAME
                        Model file
  --hpol_filter_length_dist HPOL_FILTER_LENGTH_DIST HPOL_FILTER_LENGTH_DIST
                        Length and distance to the hpol run to mark
  --runs_file RUNS_FILE
                        Homopolymer runs file
  --blacklist BLACKLIST
                        Blacklist file
  --blacklist_cg_insertions
                        Should CCG/GGC insertions be filtered out?
  --reference_file REFERENCE_FILE
                        Indexed reference FASTA file
  --output_file OUTPUT_FILE
                        Output VCF file
  --is_mutect           Is the input a result of mutect
  --flow_order FLOW_ORDER
                        Sequencing flow order (4 cycle)
  --annotate_intervals ANNOTATE_INTERVALS
                        interval files for annotation (multiple possible)
```
