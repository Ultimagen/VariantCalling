# SEC - systematic error correction using cohort noise profiles
## Environment
- conda - genomics.py3
- export PYTHONPATH="$PYTHONPATH:your/path/to/BioinfoResearch/src/python/error_correction"

## Correct systematic errors given trained model
```
python ugvc/pipelines/sec/correct_systematic_errors.py \
--model <conditional_allele_distribution.pickle> \
--gvcf <indexed vcf.gz or gvcf.gz file> \
--output_file <out.bed or out.vcf or out.pkl> \
--relevant_coords <blacklist.bed>
```

## Run training and validation pipeline
```
python ~/src/BioinfoResearch/src/python/error_correction/systematic_error_correction/pipeline.py \
 --inputs_table hap_map2_inputs.csv \
 --out_dir exome_hapmap2.1_2s_blacklist/ \
 --relevant_coords blacklists/blacklist_hapmap2.1_2s_exome.bed \
 --ground_truth_vcf ground_truth/140_samples_NYGenomes.blacklist_hapmap2.1_2s_exome.vcf.gz.nodup.vcf.gz
```
