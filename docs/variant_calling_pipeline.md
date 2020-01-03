# Pipeline to use bwa alignments and variant calling 

## Installation

1. Clone my git repository (e.g. to `/home/ec2-user/software/VariantCalling/`)
```
  cd /home/ec2-user/software/
  git clone git@github.com:Ultimagen/VariantCalling.git
  or:
  git clone https://github.com/Ultimagen/VariantCalling
```
1.1 Clone recalibration repository and buid
2. Create conda environment `conda env create -f /home/ec2-user/software/VariantCalling/setup/environment.yml`
(the path should be the right path in the computer).
3. Activate environment `conda activate genomics.py3` (or `source activate genomics.py3`)
4. Copy Broad references bucket locally (e.g. to `/data/genomes/`)

`gsutil -m rsync -x "$(gsutil ls gs://genomics-public-data/references/hg38/v0/ | awk -F '/' '{print $7 }' | grep -v 'Homo_sapiens_assembly38' | tr '\n' '|' | sed 's/|$//')" gs://genomics-public-data/references/hg38/v0/ /data/genomes/`

5. Copy helper files for variant calling
`aws s3 sync s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/ /data/genomes/broad-references/hg38/concordance/`

6. Copy additional files to the location of the genome: 

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/Homo_sapiens_assembly38.fasta.dict 
 /data/genomes/broad-references/hg38/v0/`

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/Homo_sapiens_assembly38.fasta.sizes 
/data/genomes/broad-references/hg38/v0/`

7. Download the latest gatk JAR, currently `gatk-package-ultima-v0.2-12-g4e6ad70-SNAPSHOT-local.jar`

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/jar/gatk-package-ultima-v0.2-12-g4e6ad70-SNAPSHOT-local.jar $HOME/software/gatk/`

8 Compile and install the recalibration code from `recalibration` repo. Assume into `$HOME/software/recalibration`
### Configuration file
* Create a file with list of chromosomes to run concordance on. For example
```
cat > /home/ec2-user/proj/VariantCalling/work/191018/chromosomes
chr9
chr20 
```
* Locate re-trained recalibration model (`recalibration.h5`)

* Create config file (`variant_calling.config`) of the following form: 
```
em_vc_demux_file=/home/ec2-user/proj/work/191015/420159_1p.demux.bam
em_vc_genome=/data/genomes/broad-references/hg38/v0/Homo_sapiens_assembly38.fasta
em_vc_output_dir=/home/ec2-user/proj/VariantCalling/work/191015/vc
em_vc_number_to_sample=20000000 # Set -1 if the input file is already sampled
em_vc_number_of_cpus=50
em_vc_chromosomes_list=/home/ec2-user/proj/VariantCalling/work/191018/chromosomes
em_vc_recalibration_model=/home/ec2-user/proj/VariantCalling/work/191018/recalibration.h5

em_vc_ground_truth=/data/genomes/broad-references/hg38/concordance/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf.gz
em_vc_ground_truth_highconf=/data/genomes/broad-references/hg38/concordance/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed
em_vc_gaps_hmers_filter=/data/genomes/broad-references/hg38/concordance/runs.bed
```

Optionally, this could be a section in a general config file with header 
`[EM_VC]`

### Run
```
cd /home/ec2-user/proj/work/191018/
conda activate genomics.py3
export PATH=$HOME/software/recalibration:$PATH
export PYTHONPATH=/home/ec2-user/software/VariantCalling/src/:$PYTHONPATH
export GATK_LOCAL_JAR=$HOME/software/gatk/gatk-package-ultima-v0.2-12-g4e6ad70-SNAPSHOT-local.jar
python /home/ubuntu/software/VariantCalling/src/python/pipelines/vc_pipeline.py -c variant_calling.config
```

### Output
`h5` files called chrXX.h5 that contain `concordance` and `results` dataframes. Precision and recall can be extracted from the results dataframe
