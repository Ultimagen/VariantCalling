# Variant Filtering package

## Prerequisites

1. Machine with a large memory footprint (32G)
2. Conda installation
3. Indexed (fai) fasta reference

## Installation

1. Copy the `software` directory to your machine (e.g. to `software/variantFiltering/`)
2. We recommend installing the package into a clean conda environment defined by `software/variantFiltering/setup/environment.yml`unning 
3. Create conda environment. This will create an environment called `variant.filtering`
```
  conda env create -f software/variantFiltering/setup/environment.yml
```
4. Activate this environment
```
  conda activate variant.filtering
```
5. Install the package 
```
  pip install variantFiltering
```

## Running the filtering pipeline

In this example we are filtering *indexed* VCF file test.vcf.gz

1. Remove unused and purely symbolic alleles
```
  gatk SelectVariants -V test.vcf.gz -O test.tmp.vcf.gz  --remove-unused-alternates
  gatk SelectVariants -V test.tmp.vcf.gz -O test.filter.vcf.gz  --exclude-non-variants \
          --select-type-to-exclude SYMBOLIC
```
2. Filter and score the variants 
```
  filter_variants_pipeline.py  --input_file test.filter.vcf.gz \
  --model_file software/variantFiltering/data/model.pkl \
  --model_name threshold_model_ignore_gt_excl_hpol_runs \
  --runs_file data/runs.bed \
  --reference_file Homo_sapiens_assembly38.fasta \
  --output_file test.final.vcf.gz
```

## Notes

1. The following files are provided
 - filtering models `variantFiltering/data/model.pkl`
 - homopolymer runs file (we annotate variants close or inside homopolymer runs longer than 10 bases 
 by HPOL_RUN in the VCF filter). 
 Homopolymer annotation of hg38 genome is provided in `variantFiltering/data/runs.bed`. Create an 
 own bed file if other genomes are used

2. In case high memory machne is not available it is suggested to split the VCF file into smaller blocks
   and perform filtering block-by-block
   
3. The filtering pipeline adds an info field `TREE_SCORE` that contains the confidence score of the variant 
(1 - high). Low quality variants are also labeled `LOW_SCORE` in the `FILTER` column. 

4. We provide two flavors of the filtering models that perform similarly in our experience: based on a simple 
thresholding on info fields and a decision tree model. The model can be selected by `model_name` parameter
and the names are `threshold_model_ignore_gt_excl_hpol_runs` or `dt_model_ignore_gt_excl_hpol_runs`. 
