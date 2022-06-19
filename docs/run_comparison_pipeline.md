## run_comparison_pipeline.py

This script receives UG callset and a ground truth calset and compares them using VCFEVAL as
the internal evaluation pipeline. The output of the process is HDF5 file split by chromosome
that contains for each variant classification (FP/TP/FN), type (hmer-indel, non-hmer indel, SNP),
local motif etc.

### Usage
```
usage: run_comparison_pipeline.py [-h] --n_parts N_PARTS --input_prefix
                                  INPUT_PREFIX --output_file OUTPUT_FILE
                                  --output_interval OUTPUT_INTERVAL --gtr_vcf
                                  GTR_VCF [--cmp_intervals CMP_INTERVALS]
                                  --highconf_intervals HIGHCONF_INTERVALS
                                  [--runs_intervals RUNS_INTERVALS]
                                  [--annotate_intervals ANNOTATE_INTERVALS]
                                  --reference REFERENCE
                                  [--reference_dict REFERENCE_DICT]
                                  [--coverage_bw_high_quality COVERAGE_BW_HIGH_QUALITY]
                                  [--coverage_bw_all_quality COVERAGE_BW_ALL_QUALITY]
                                  --call_sample_name CALL_SAMPLE_NAME
                                  --truth_sample_name TRUTH_SAMPLE_NAME
                                  [--header_file HEADER_FILE] [--filter_runs]
                                  [--hpol_filter_length_dist HPOL_FILTER_LENGTH_DIST HPOL_FILTER_LENGTH_DIST]
                                  [--ignore_filter_status]
                                  [--flow_order FLOW_ORDER]
                                  [--output_suffix OUTPUT_SUFFIX]
                                  [--concordance_tool CONCORDANCE_TOOL]
                                  [--disable_reinterpretation] [--is_mutect]
                                  [--n_jobs N_JOBS] [--verbosity VERBOSITY]

Compare VCF to ground truth

optional arguments:
  -h, --help            show this help message and exit
  --n_parts N_PARTS     Number of parts that the VCF is split into
  --input_prefix INPUT_PREFIX
                        Prefix of the input file
  --output_file OUTPUT_FILE
                        Output h5 file
  --output_interval OUTPUT_INTERVAL
                        Output bed file of intersected intervals
  --gtr_vcf GTR_VCF     Ground truth VCF file
  --cmp_intervals CMP_INTERVALS
                        Ranges on which to perform comparison
                        (bed/interval_list)
  --highconf_intervals HIGHCONF_INTERVALS
                        High confidence intervals (bed/interval_list)
  --runs_intervals RUNS_INTERVALS
                        Runs intervals (bed/interval_list)
  --annotate_intervals ANNOTATE_INTERVALS
                        interval files for annotation (multiple possible)
  --reference REFERENCE
                        Reference genome
  --reference_dict REFERENCE_DICT
                        Reference genome dictionary
  --coverage_bw_high_quality COVERAGE_BW_HIGH_QUALITY
                        BigWig file with coverage only on high mapq reads
  --coverage_bw_all_quality COVERAGE_BW_ALL_QUALITY
                        BigWig file with coverage on all mapq reads
  --call_sample_name CALL_SAMPLE_NAME
                        Name of the call sample
  --truth_sample_name TRUTH_SAMPLE_NAME
                        Name of the truth sample
  --header_file HEADER_FILE
                        Desired header
  --filter_runs         Should variants on hmer runs be filtered out
  --hpol_filter_length_dist HPOL_FILTER_LENGTH_DIST HPOL_FILTER_LENGTH_DIST
                        Length and distance to the hpol run to mark
  --ignore_filter_status
                        Ignore variant filter status
  --flow_order FLOW_ORDER
                        Sequencing flow order (4 cycle)
  --output_suffix OUTPUT_SUFFIX
                        Add suffix to the output file
  --concordance_tool CONCORDANCE_TOOL
                        The concordance method to use (GC or VCFEVAL)
  --disable_reinterpretation
                        Should re-interpretation be run
  --is_mutect           Are the VCFs output of Mutect (false)
  --n_jobs N_JOBS       n_jobs of parallel on contigs
  --verbosity VERBOSITY
                        Verbosity: ERROR, WARNING, INFO, DEBUG
```

### Example
```
run_comparison_pipeline.py \
--n_parts 0 --input_prefix 002850-UGAv3-2_40x.filtered \
--output_file 002850-UGAv3-2_40x.h5 \
--output_interval output_interval.bed \
--gtr_vcf HG002_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.vcf.gz \
--highconf_intervals HG001_GRCh38_1_22_v4.2.1_benchmark.bed \
--reference Homo_sapiens_assembly38.fasta \
--runs_intervals runs.conservative.bed \
--call_sample_name UGAv3-2 \
--truth_sample_name HG002 \
--disable_reinterpretation \
--ignore_filter_status \
--annotate_intervals LCR-hs38.bed \
--n_jobs 8
```

### See also
[Evaluation of UG callsets](docs/howto-evaluate-ug-callset.md)
