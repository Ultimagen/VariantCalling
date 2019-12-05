# Pipeline to calculate coverage and duplication rate using rapid QC

## Installation
1. Clone my git repository (e.g. to `/home/ubuntu/software/VariantCalling/`)
```
  cd /home/ubuntu/software/
  git clone git@github.com:Ultimagen/VariantCalling.git
  or:
  git clone https://github.com/Ultimagen/VariantCalling
```
2. Create conda environment
`conda env create -f /home/ubuntu/software/VariantCalling/setup/environment.yml`
(the path should be the right path in the computer).

**Note:** Update the environment if it exists 
`conda env update -f /home/ubuntu/software/VariantCalling/setup/environment.yml`. 

3. Activate environment 
`conda activate genomics.py3` 
(or `source activate genomics.py3`)

4. Copy Broad references bucket locally (e.g. to `/data/genomes/`)

`aws s3 sync --exclude '*' --include "Homo_sapiens_assembly38*" s3://broad-references/hg38/v0/ /data/genomes/broad-references/hg38/v0/`

5. Copy evaluation intervals
```
aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/chr9.hg38.eval.interval_list /data/genomes/broad-references/hg38/v0/
aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/chr9.hg38.exome.eval.interval_list /data/genomes/broad-references/hg38/v0/
```


### Configuration file

Create an intervals file (`rapid_qc.intervals`) that contains tab-separated lines of name, filename of intervals. E.g.

```
genome /data/genomes/broad-references/hg38/v0/chr9.hg38.eval.interval_list
exome /data/genomes/broad-references/hg38/v0/chr9.hg38.exome.eval.interval_list
```

Create config file (`rapid_qc.config`) of the following form: 

```
[rapid_params]
rqc_demux_file=/home/ubuntu/proj/VariantCallig/work/191015/420159_1p.demux.bam 
em_vc_genome=/data/genomes/broad-references/hg38/v0/Homo_sapiens_assembly38.fasta
em_vc_output_dir=/home/ubuntu/proj/VariantCalling/work/191015/em
em_vc_number_of_cpus=40
rqc_chromosome=chr9 #or other chromosome as you see fit
rqc_evaluation_intervals=/home/ubuntu/proj/VariantCalling/work/191128/rapid_qc.intervals

[rapid_intervals]
names = [genome,exome]
paths = 
genome = /data/genomes/broad-references/hg38/v0/chr9.hg38.eval.interval_list
exome = /data/genomes/broad-references/hg38/v0/chr9.hg38.exome.eval.interval_list

```

Optionally, this could be a section in a general config file with header. Note that for this pipeline one needs to use the _unsampled_ BAM. 


### Run
```
cd /home/ubuntu/proj/work/191015/
conda activate genomics.py3
python /home/ubuntu/software/VariantCalling/src/python/pipelines/rapid_qc_pipeline.py -c rapid_qc.config
```

### Output
Text output files will be named: 
 - `.rmdup.metrics` - duplication metrics
 - `.coverage.metrics` - coverage metrics
