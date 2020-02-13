# Filering and comparing the results of variant calling

## Installation

1. Clone my git repository (e.g. to `/home/ec2-user/software/VariantCalling/`)
    ```
    cd /home/ec2-user/software/
    git clone git@github.com:Ultimagen/VariantCalling.git
    or:
    git clone https://github.com/Ultimagen/VariantCalling
    ```

3. Create conda environment `conda env create -f /home/ec2-user/software/VariantCalling/setup/environment.yml`
(the path should be the right path in the computer, if the environment already exists - run `conda env update -f /home/ec2-user/software/VariantCalling/setup/environment.yml`).

4. Activate environment `conda activate genomics.py3` (or `source activate genomics.py3`)

5. Copy Broad references bucket locally (e.g. to `/data/genomes/`)

`gsutil -m rsync -x "$(gsutil ls gs://gcp-public-data--broad-references/hg38/v0/ | awk -F '/' '{print $6 }' | grep -v 'Homo_sapiens_assembly38' | tr '\n' '|' | sed 's/|$//')" gs://gcp-public-data--broad-references/hg38/v0/ /data/genomes/broad-references/hg38/v0/`

6. Copy helper files for variant calling
`aws s3 sync s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/ /data/genomes/broad-references/hg38/concordance/`

7. Copy additional files to the location of the genome: 

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/Homo_sapiens_assembly38.fasta.dict 
 /data/genomes/broad-references/hg38/v0/`

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/Homo_sapiens_assembly38.fasta.sizes 
/data/genomes/broad-references/hg38/v0/`

## Workflow options

### Generate concordance dataframe for downstream investigation

`GATK VCF, ground truth VCF -> run_comparison_pipeline.py -> output.h5`

### Train filtering model

`GATK VCF, ground truth VCF -> run_comparison_pipeline.py -> output.h5 -> train_models_pipeline.py -> models.pkl`

### Filter variant calling set and compare to ground truth

`GATK VCF, models.pkl -> filter_variants_pipeline.py -> output.vcf, ground truth VCF -> run_comparison_pipeline.py -> output.h5 -> train_models_pipeline.py -> models.pkl`

## Step descriptions

**Note:** Before running the scripts do
`export PYTHONPATH=<REPLACE WITH PATH TO VARIANTCALLING REPO>/src/:$PYTHONPATH`

### `run_comparison_pipeline.py`

Generates genotype concordance dataframe between the the variant call set and the ground truth set. The dataframe is saved in HDF5 file

```
usage: run_comparison_pipeline.py [-h] --n_parts N_PARTS --input_prefix
                                   INPUT_PREFIX --output_file OUTPUT_FILE
                                   --gtr_vcf GTR_VCF
                                   --cmp_intervals CMP_INTERVALS
                                   --highconf_intervals HIGHCONF_INTERVALS
                                   [--runs_intervals RUNS_INTERVALS]
                                   --reference REFERENCE 
                                   [--aligned_bam ALIGNED_BAM]
                                   --call_sample_name
                                   CALL_SAMPLE_NAME --truth_sample_name
                                   TRUTH_SAMPLE_NAME [--find_thresholds]
                                   [--filter_runs]

Arguments:
  -h, --help            show this help message and exit
  --n_parts N_PARTS     Number of parts that the VCF is split into (0 if none)
  --input_prefix INPUT_PREFIX
                        Prefix of the input file (until .vcf.gz)
  --output_file OUTPUT_FILE
                        Output h5 file
  --gtr_vcf GTR_VCF     Ground truth VCF file
  --cmp_intervals CMP_INTERVALS
                        Ranges on which to perform comparison
  --highconf_intervals HIGHCONF_INTERVALS
                        High confidence intervals 
  --runs_intervals RUNS_INTERVALS
                        Runs intervals (locations of homopolymer repeats)
  --reference REFERENCE
                        Reference genome
  --aligned_bam ALIGNED_BAM
                        Aligned bam (optional, do not use on whole genome VCF)
  --call_sample_name CALL_SAMPLE_NAME
                        Name of the call sample
  --truth_sample_name TRUTH_SAMPLE_NAME
                        Name of the truth sample
  --find_thresholds     Should precision recall thresholds be found
  --filter_runs         Should variants on hmer runs be filtered out in the output

```

### Example file locations

```
GTR_VCF=data/genomes/broad-references/hg38/v0/concordance/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.broad-header.vcf.gz
HIGHCONF_INTERVALS=/data/genomes/broad-references/hg38/v0/concordance/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed
RUNS_INTERVALS=/data/genomes/broad-references/hg38/v0/concordance/runs.bed
CMP_INTERVALS=/data/genomes/broad-references/hg38/v0/concordance/chr20.hg38.comparison.interval_list
REFERENCE=/data/genomes/broad-references/hg38/v0/Homo_sapiens_assembly38.fasta
CALL_SAMPLE_NAME=sm1 # default sample name in our data, change as appropriate
TRUTH_SAMPLE=INTEGRATION # sample name in GIAB dataset
```

### `filter_variants_pipeline.py`

Filters VCF file

```
usage: filter_variants_pipeline.py [-h] --input_file INPUT_FILE --model_file
                                   MODEL_FILE --model_name MODEL_NAME
                                   --runs_file RUNS_FILE --reference_file
                                   REFERENCE_FILE --output_file OUTPUT_FILE

Filter VCF

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Name of the input VCF file
  --model_file MODEL_FILE
                        Pickle model file
  --model_name MODEL_NAME
                        Model file
  --runs_file RUNS_FILE
                        Homopolymer runs file
  --reference_file REFERENCE_FILE
                        Indexed reference FASTA file
  --output_file OUTPUT_FILE
                        Output VCF file

```

**Note:** model_file is a dictionary with multiple fields

Models: 

* `threshold_model`, `dt_model` - thresholding on QUAL/SOR or decision tree model
* `ignore_gt` versus `include_gt` - false positive, false negative, true positive are determined only on the variants or on the alleles
* `incl_hpol_runs` versus `excl_hpol_runs` - are variants close or inside homopolymer runs used or not
* `recall_precision` - tuple of _optimal_ recall/precision for the model
* `recall_precision_curve` - recall precision curve of the model

```
threshold_model_ignore_gt_incl_hpol_runs
threshold_model_recall_precision_ignore_gt_incl_hpol_runs
threshold_model_precision_recall_curve_ignore_gt_incl_hpol_runs
threshold_model_include_gt_incl_hpol_runs
threshold_model_recall_precision_include_gt_incl_hpol_runs
threshold_model_precision_recall_curve_include_gt_incl_hpol_runs
threshold_model_ignore_gt_excl_hpol_runs
threshold_model_recall_precision_ignore_gt_excl_hpol_runs
threshold_model_precision_recall_curve_ignore_gt_excl_hpol_runs
threshold_model_include_gt_excl_hpol_runs
threshold_model_recall_precision_include_gt_excl_hpol_runs
threshold_model_precision_recall_curve_include_gt_excl_hpol_runs
dt_model_ignore_gt_incl_hpol_runs
dt_model_recall_precision_ignore_gt_incl_hpol_runs
dt_model_recall_precision_curve_ignore_gt_incl_hpol_runs
dt_model_include_gt_incl_hpol_runs
dt_model_recall_precision_include_gt_incl_hpol_runs
dt_model_recall_precision_curve_include_gt_incl_hpol_runs
dt_model_ignore_gt_excl_hpol_runs
dt_model_recall_precision_ignore_gt_excl_hpol_runs
dt_model_recall_precision_curve_ignore_gt_excl_hpol_runs
dt_model_include_gt_excl_hpol_runs
dt_model_recall_precision_include_gt_excl_hpol_runs
dt_model_recall_precision_curve_include_gt_excl_hpol_runs
```

### `train_models_pipeline.py`

Trains models on HDF5 dataframe. The output models are saved in pickle file. The usage is self-explanatory.

```
usage: train_models_pipeline.py [-h]
                                (--input_file INPUT_FILE | --input_fofn INPUT_FOFN)
                                --output_file OUTPUT_FILE

Train filtering models on the concordance file

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Name of the input h5 file
  --input_fofn INPUT_FOFN
                        Input file containing list of the h5 file names to
                        concatenate
  --output_file OUTPUT_FILE
                        Output pkl file
```

