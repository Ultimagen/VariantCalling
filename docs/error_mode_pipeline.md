# Pipeline to calculate error rate using bwa alignments

## Installation
1. Clone my git repository (e.g. to `/home/ubuntu/software/VariantCalling/`)
```
  cd /home/ubuntu/software/
  git clone git@github.com:Ultimagen/VariantCalling.git
```
2. Create conda environment `conda env create -f /home/ubuntu/software/VariantCalling/setup/environment.yml`
3. Activate environment `conda activate genomics.py3`
4. Copy Broad references bucket locally (e.g. to `/data/genomes/`)

`aws s3 sync s3://broad-references/hg19/ /data/genomes/broad-references/hg19/`



### Configuration file
Create config file (`error_metrics.config`) of the following form: 

```
em_vc_demux_file=/home/ubuntu/proj/work/191015/420159_1p.demux.bam
em_vc_genome=/data/genomes/broad-references/hg19/v0/Homo_sapiens_assembly19.fasta
em_vc_output_dir=/home/ubuntu/proj/VariantCalling/work/191015/em
em_vc_number_to_sample=20000000 # Set -1 if the input file is already sampled
```

Optionally, this could be a section in a general config file with header 
`[EM_VC]`

### Run
```
/home/ubuntu/proj/work/191015/
conda activate genomics.py3
python /home/ubuntu/software/VariantCalling/src/python/pipelines/error_rate_metrics_pipeline.py -c error_metrics.config
```

### Output
Text output files will be named: 
`.sort.metrics` - error metrics for unfiltered aligned data
`.sort.filter.metrics` - error metrics for filtered (>Q20) aligned data
`.idxstats` - alignment statistics

All outputs will be concatenated into HDF5 file `metrics.h5` with keys `bwa_alignment_stats` and `bwa_error_rates`

