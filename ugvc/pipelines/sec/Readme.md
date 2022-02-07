# Environment
- conda - genomics.py3
- export PYTHONPATH="$PYTHONPATH:your/path/to/BioinfoResearch/src/python/error_correction"

# Correct systematic errors given trained model
```
python src/python/error_correction/systematic_error_correction/correct_systematic_errors.py --model <conditional_allele_distribution.pickle> --gvcf <indexed vcf.gz or gvcf.gz file> --output_file <out.bed or out.vcf> --relevant_coords <blacklist.bed>
```
# Preparation

Query and download as csv the hap-map2 table from Tableu:
- http://162.222.176.194/#/views/HapMap/HapMapTable?:iid=3

Extract unique workflow-ids from csv file:
```
awk -F , '{print $4}' /mnt/c/Users/doron/Documents/error_correction/HapMap_Table_data.csv | uniq > hapmap2_workflow_ids.csv
```

Extract paths to gvcf files using cromshell on zbroad server:
```
cat doronst/hapmap2_workflow_ids.csv | awk '{print "cromshell -t 30 metadata "$1" | jq \047.outputs.\"JukeboxVC.output_gvcf\"\047"}' | sh > doronst/hapmap2_gvcf_files.txt
```

Extract sample_ids using cromshell on zbroad server:
```
cat doronst/hapmap2_workflow_ids.csv | awk '{print "cromshell -t 30 metadata "$1" | jq \047.inputs.\"gt_right_sample\"\047"}' | sh > doronst/hapmap2_right_sample_ids.txt
```

Download blacklist
```
# hg001 10-samples exome blacklist
gsutil cp gs://ultimagen-maya/Notebooks/mutect/data_simulation/common_vars_clean_10.bed blacklists/blacklist_hg001_10s_exome.bed

# hapmap2.1 2-samples exome blacklist 
gsutil cp gs://ultimagen-maya/Notebooks/mutect/data_simulation/common_vars_hapmap_2_1_exome_allele_1_again.bed blacklists/blacklist_hapmap2.1_2s_exome.bed

# hapmap2.1 2-samples chr9 blacklist
gsutil cp gs://ultimagen-maya/Notebooks/mutect/data_simulation/common_vars_hapmap_2_1_chr9_allele_1_again.bed blacklists/blacklist_hapmap2.1_2s_chr9.bed
```

Extract blacklist position from ground-truth files
```
# hg001 10-samples exome blacklist
nohup bcftools view gs://ultimagen-ilya-new/VariantCalling/work/211101/140_samples_NYGenomes.vcf.gz -R blacklist_v2.sort.bed -Oz > 140_samples_NYGenomes.exome_black_list_v2.vcf.gz &

# hapmap2.1 2-samples exome blacklist
nohup bcftools view gs://ultimagen-ilya-new/VariantCalling/work/211101/140_samples_NYGenomes.vcf.gz -Oz | bedtools intersect -a stdin -b blacklists/blacklist_hapmap2.1_2s_exome.bed -header > ground_truth/140_samples_NYGenomes.blacklist_hapmap2.1_2s_exome.vcf &

# hapmap2.1 2-samples chr9 blacklist
nohup bcftools view gs://ultimagen-ilya-new/VariantCalling/work/211101/140_samples_NYGenomes.vcf.gz chr9 -Oz | bedtools intersect -a stdin -b blacklists/blacklist_hapmap2.1_2s_chr9.bed -header > ground_truth/140_samples_NYGenomes.blacklist_hapmap2.1_2s_chr9.vcf.gz &

# hapmap2.1 2-samples wgs blacklist
nohup bcftools view gs://ultimagen-ilya-new/VariantCalling/work/211101/140_samples_NYGenomes.vcf.gz -Oz | bedtools intersect -a stdin -b blacklists/blacklist_hapmap2.1_2s_chr9.bed -header | uniq > ground_truth/140_samples_NYGenomes.blacklist_hapmap2.1_2s_wgs.vcf.gz &

# hg002 chr9 blacklist
bcftools view gs://concordance/ground-truths-files/HG002_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.vcf.gz chr9 | bedtools intersect -a stdin -b blacklists/blacklist_hapmap2.1_2s_chr9.bed -header > HG002_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.blacklist_hapmap2.1_2s_chr9.vcf.gz
bcftools view gs://concordance/ground-truths-files/HG002_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.vcf.gz -R blacklists/blacklist_hapmap2.1_2s_exome.bed -Oz > ground_truth/HG002_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.blacklist_hapmap2.1_2s_exome.vcf.gz

# hapmap2.1 2-samples chr9 redlist
bcftools view gs://ultimagen-ilya-new/VariantCalling/work/211101/140_samples_NYGenomes.vcf.gz chr9 -Oz | bedtools intersect -a stdin -b redlists/discordant_variants_2s.bed -header > ground_truth/140_samples_NYGenomes.redlist_hapmap2.1_2s_chr9.vcf.gz &   
```

Filter blacklist to contain only positions besides hmers.
```
grep -v AS_HmerLength=0 chr9_hapmap2.1_2s_blacklist/gvcf/HG00239.g.vcf | grep -v "#" | awk '{print $1"\t"$2-1"\t"$2}' > blacklists/blacklist_hapmap2.1_2s_chr9.hg00239_hmer.bed
cat chr9_hapmap2.1_2s_blacklist/gvcf/*.g.vcf | grep --color=auto AS_HmerLength | grep --color=auto -v AS_HmerLength=0 | awk '{print $1,$2-1,$2}' | sort -k 2,2n | uniq > blacklists/blacklist_hapmap2.1_2s_chr9.hmer.bed
```

Filter blacklist to contain only isolated positions
```
bedtools window -a blacklists/blacklist_hapmap2.1_2s_chr9.bed -b blacklists/blacklist_hapmap2.1_2s_chr9.bed -w 100 | awk '{print $1,$2,$3}' | uniq -c | awk '$1<5 {print $2"\t"$3"\t"$4}' > blacklists/blacklist_hapmap2.1_2s_chr9.isolated.bed
```

Remove duplicates from bcf_tools output
```
# hg001 10-samples exome blacklist
sh scripts/remove_vcf_duplicates.sh 140_samples_NYGenomes.exome_black_list_v2.vcf.gz

# hapmap2.1 2-samples exome blacklist
sh scripts/remove_vcf_duplicates.sh ground_truth/140_samples_NYGenomes.blacklist_hapmap2.1_2s_exome.vcf.gz
```

Run training and validation pipeline
```
python ~/src/BioinfoResearch/src/python/error_correction/systematic_error_correction/pipeline.py \
 --inputs_table hap_map2_inputs.csv \
 --out_dir exome_hapmap2.1_2s_blacklist/ \
 --relevant_coords blacklists/blacklist_hapmap2.1_2s_exome.bed \
 --ground_truth_vcf ground_truth/140_samples_NYGenomes.blacklist_hapmap2.1_2s_exome.vcf.gz.nodup.vcf.gz
```

Create HCR with high coverage and mappability:
```
bedtools subtract -a hcr/HG002_HG001_intersect_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed -b hcr/low_coverage_or_mappability.bed > hcr/hcr_regions.bed
```