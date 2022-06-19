# SEC - systematic error correction using cohort noise profiles
## Environment
- conda - genomics.py3

## Correct systematic errors given trained model
```
python ugvc correct_systematic_errors \
--model <conditional_allele_distribution.pickle> \
--gvcf <indexed vcf.gz or gvcf.gz file> \
--output_file <out.bed or out.vcf or out.pkl> \
--relevant_coords <blacklist.bed>
```
```
