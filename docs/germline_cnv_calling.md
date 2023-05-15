# Germline cnv calling tools
The following scripts are used for germline cnv calling for a given sample/cohort.

### Assumptions: 
cn.mops conda enviorment is installed. 

### UG specific Files might be required for the analysis (download locally):
* CNV-LCR: gs://concordanz/hg38/UG-High-Confidence-Regions/v*/ug_cnv_lcr.bed
* Hapmap precalucalted cohort: gs://concordanz/hg38/HapMap2_210samples_cohort.ReadsCount.rds

## Scripts:
### get_reads_count_from_bam.R
Calculate reads count per window for a given bam file (with its corresponding index file).
```
usage: get_reads_count_from_bam.R [-h] [-i INPUT_BAM_FILE]
                                  [-refseq REFSEQNAMES_STRING]
                                  [-wl WINDOW_LENGTH] [-p PARALLEL]
                                  [-o BASE_FILE_NAME] [--save_hdf]

options:
  -h, --help            show this help message and exit
  -i INPUT_BAM_FILE, --input_bam_file INPUT_BAM_FILE
                        input bam file path
  -refseq REFSEQNAMES_STRING, --refSeqNames_string REFSEQNAMES_STRING
                        chromosome names in comma seperated format. e.g,
                        chr1,chr2,chrX
  -wl WINDOW_LENGTH, --window_length WINDOW_LENGTH
                        window length (#bp) for which reads count is
                        calculated for
  -p PARALLEL, --parallel PARALLEL
                        number of parallel processes
  -o BASE_FILE_NAME, --base_file_name BASE_FILE_NAME
                        out base file name
  --save_hdf            whether to save reads count data-frames in hdf5 format
```
Example:
```
conda activate cn.mops
Rscript --vanilla  /VariantCalling/ugvc/cnv/get_reads_count_from_bam.R \
    -i 002494-X0033.MAPQ0.bam \
    -refseq chr1 \
    -wl 1000 \
    -p 30 \
    -o 002494-X0033
```
output is a GenomicRanges object consists of reads counts per window. 

### merge_reads_count_sample_to_cohort.R
Add reads count of a single sample to a cohort's reads count object.
```
usage: merge_reads_count_sample_to_cohort.R [-h]
                                            [-cohort_rc COHORT_READS_COUNT_FILE]
                                            [-sample_rc SAMPLE_READS_COUNT_FILE]
                                            [--save_hdf]

options:
  -h, --help            show this help message and exit
  -cohort_rc COHORT_READS_COUNT_FILE, --cohort_reads_count_file COHORT_READS_COUNT_FILE
                        input cohort reads count file in rds format
  -sample_rc SAMPLE_READS_COUNT_FILE, --sample_reads_count_file SAMPLE_READS_COUNT_FILE
                        input sample reads count file in rds format
  --save_hdf            whether to save reads count data-frames in hdf5 format
```
Example:
```
conda activate cn.mops
Rscript --vanilla /VariantCalling/ugvc/cnv/merge_reads_count_sample_to_cohort.R \
    -cohort_rc test_cohort_reads_count.rds \
    -sample_rc test_sample.ReadCounts.rds
```
output is a single GenomicRanges object consists of reads counts per window for all cohort's samples + input sample.

### create_reads_count_cohort_matrix.R
Combine reads counts of several input samples into a single GenomicRanges object.
```
usage: create_reads_count_cohort_matrix.R [-h]
                                          [-samples_read_count_files_list SAMPLES_READ_COUNT_FILES_LIST]
                                          [--save_csv] [--save_hdf]

options:
  -h, --help            show this help message and exit
  -samples_read_count_files_list SAMPLES_READ_COUNT_FILES_LIST, --samples_read_count_files_list SAMPLES_READ_COUNT_FILES_LIST
                        file containing a list of all reads count rds files
                        per sample
  --save_csv            whether to save reads count data-frames in csv format
  --save_hdf            whether to save reads count data-frames in hdf5 format
```
Example:
```
conda activate cn.mops
Rscript --vanilla /VariantCalling/ugvc/cnv/create_reads_count_cohort_matrix.R \
    -samples_read_count_files_list input_read_count_files_list.txt
```
output is a single GenomicRanges object consists of reads counts per window for all input samples.

### cnv_calling_using_cnmops.R
Germline CNV calling for a given cohort's reads count.
CNV calling is done using cn.mops algorithm documented here : https://academic.oup.com/nar/article/40/9/e69/1136601
```
usage: cnv_calling_using_cnmops.R [-h] [-cohort_rc COHORT_READS_COUNT_FILE]
                                  [-minWidth MIN_WIDTH_VAL] [-p PARALLEL]
                                  [--save_hdf] [--save_csv]

options:
  -h, --help            show this help message and exit
  -cohort_rc COHORT_READS_COUNT_FILE, --cohort_reads_count_file COHORT_READS_COUNT_FILE
                        input cohort reads count file in Rdata format
  -minWidth MIN_WIDTH_VAL, --min_width_val MIN_WIDTH_VAL
  -p PARALLEL, --parallel PARALLEL
                        number of parallel processes
  --save_hdf            whether to save additional data-frames in hdf5 format
  --save_csv            whether to save additional data-frames in csv format
```
Example:
```
Rscript --vanilla /VariantCalling/ugvc/cnv/cnv_calling_using_cnmops.R \
    -cohort_rc merged_cohort_reads_count.rds \
    -minWidth 2 \
    -p 30 \
```
output is a csv file consisting of all called CNVs for all samples in the cohort. 

### filter_sample_cnvs.py
Filter CNV calls by length and UG-CNV-LCR regions.
### Usage
```
usage: filter_sample_cnvs.py [-h] --input_bed_file INPUT_BED_FILE
                             --intersection_cutoff INTERSECTION_CUTOFF
                             --cnv_lcr_file CNV_LCR_FILE --min_cnv_length
                             MIN_CNV_LENGTH [--out_directory OUT_DIRECTORY]
                             [--verbosity VERBOSITY]

Filter cnvs bed file by: ug_cnv_lcr, length

optional arguments:
  -h, --help            show this help message and exit
  --input_bed_file INPUT_BED_FILE
                        input bed file with .bed suffix
  --intersection_cutoff INTERSECTION_CUTOFF
                        intersection cutoff for bedtools substruct function
  --cnv_lcr_file CNV_LCR_FILE
                        UG-CNV-LCR bed file
  --min_cnv_length MIN_CNV_LENGTH
  --out_directory OUT_DIRECTORY
                        out directory where intermediate and output files will
                        be saved. if not supplied all files will be written to
                        current directory
  --verbosity VERBOSITY
                        Verbosity: ERROR, WARNING, INFO, DEBUG
```
### Example
```
conda activate genomics.py3
python /VariantCalling/ugvc filter_sample_cnvs \
        --input_bed_file test_sample.cnvs.bed \
        --intersection_cutoff 0.5 \
        --cnv_lcr_file ug_cnv_lcr.bed \
        --min_cnv_length 10000;
```





