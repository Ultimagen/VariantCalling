# Post-GATK variant filtering

## Assumptions

Instructions below assume that the GATK was run using this reference file: `gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta` since multiple annotation and training files require this.

## Prerequisites

    1. Machine with a large memory footprint (32G)

    2. Conda installation

## Installation

1. Copy the VariantCalling directory to your machine (e.g. to `software/VariantCalling/`)

2. Create a clean conda environment defined by `software/VariantCalling/setup/environment.yml`

3. Create conda environment:
`conda env create -f software/VariantCalling/setup/environment.yml`
This will create an environment called `genomics.py3`

4. Install ugvc package:
```
conda activate genomics.py3
cd software/VariantCalling
pip install .
```


## Files required for the analysis (download locally)

Publicly available files

    * `gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta`

    * `gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.fai`

    * `gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.dict`

    * `gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.dbsnp138.vcf.gz`

    * `gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.dbsnp138.vcf.gz.tbi`

UG-specific files (note that the exact location might be different dependent on the data share)

    * `runs.conservative.bed` - annotation of homopolymer runs in the reference genome

    * `blacklist_ua_good_old_blacklist.h5` - list of locations prone to be false positives

    * `exome.twist.bed` - exome annotation from Twist

    * `LCR-hs38.bed` - LCR defined by Heng Li

    * `mappability.0.bed` - high mappability locations from the UCSC


## Running the filtering pipeline

The filtering pipeline has two main steps:

    * Training filtering model: in this step a small subset of known variants from the VCF is labeled as true positive and a small subset of variants from the VCF that fall on locations that tend to be problematic are labeled as false positive. Machine learning model (random forest) is then trained to distinguish between these sets

    * Filtering the VCF: the model trained in the first step is then applied to the input VCF. At this point we add a PASS/LOW_SCORE filter as well as a numerical confidence score `TREE_SCORE`. Optionally, a blacklist object can be supplied with locations that can't be trusted and these will be labeled as `COHORT_FP`.

In this example we are filtering indexed VCF file `test.vcf.gz`


### Model training

Activate the conda environment

`conda activate genomics.py3`

#### Add dbSNP annotations to the input VCF

```
    gatk --java-options "-Xms10000m" VariantAnnotator \
    -R Homo_sapiens_assembly38.fasta \
    -V test.vcf.gz \
    -O test.annotated.vcf.gz \
    --dbsnp Homo_sapiens_assembly38.dbsnp138.vcf.gz \
    -A StrandOddsRatio
```

#### Train filtering model

```
python train_models_pipeline.py \
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

Verify that this command generated a file `test.model.pkl`

### Filter VCF

Activate the conda environment: `conda activate genomics.py3`

```
filter_variants_pipeline.py \
    --input_file test.vcf.gz \
    --model_file test.model.pkl \
    --model_name rf_model_ignore_gt_incl_hpol_runs \
    --flow_order TGCA \
    --runs_file runs.conservative.bed \
    --annotate_intervals LCR-hs38.bed \
    --annotate_intervals exome.twist.bed \
    --annotate_intervals mappability.0.bed \
    --hpol_filter_length_dist 12 10 \
    --reference_file ../genomes/Homo_sapiens_assembly38.fasta \
    --output_file test.filter.vcf.gz
```

Confirm that this created a file called `test.filter.vcf.gz`.
