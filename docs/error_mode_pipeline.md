# Pipeline to calculate error rate using bwa alignments

## Installation
1. Download [environment.yml](https://github.com/Ultimagen/VariantCalling/blob/master/setup/environment.yml)
2. Create conda environment `conda env create -f environment.yml`
3. Copy Broad references bucket locally (e.g. to `/data/genomes/`)

`aws s3 sync s3://broad-references/ /data/genomes/broad-references/`

4. Clone my git repository (e.g. to `/home/ubuntu/software/VariantCalling/`)
```
  cd /home/ubuntu/software/
  git clone git@github.com:Ultimagen/VariantCalling.git
```

### Configuration file
Create config file (`error_metrics.config`) of the following form: 

```
demux_file=/home/ubuntu/proj/work/191015/420159_1p.demux.bam
genome=/data/genomes/hg19/v0/Homo_sapiens_assembly19.fasta
output_dir=/home/ubuntu/proj/VariantCalling/work/191015/em
number_to_sample=20000000 # Set -1 if the input file is already sampled
```

### Run
```
/home/ubuntu/proj/work/191015/
conda activate genomics.py3
python /home/ubuntu/software/VariantCalling/python/pipelines/error_rate_metrics_pipeline.py -c error_metrics.config
```
