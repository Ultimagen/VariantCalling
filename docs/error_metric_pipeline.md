# Pipeline to calculate error rate using bwa alignments

## Installation
1. Clone my git repository (e.g. to `/home/ubuntu/software/VariantCalling/`)
```
  cd /home/ubuntu/software/
  git clone git@github.com:Ultimagen/VariantCalling.git
  or:
  git clone https://github.com/Ultimagen/VariantCalling
```
2. Create conda environment `conda env create -f /home/ubuntu/software/VariantCalling/setup/environment.yml`
(the path should be the right path in the computer).

3. Activate environment `conda activate genomics.py3` (or `source activate genomics.py3`)

4. Copy Broad references bucket locally (e.g. to `/data/genomes/`)

`gsutil -m rsync -x "$(gsutil ls gs://gcp-public-data--broad-references/hg38/v0/ | awk -F '/' '{print $6 }' | grep -v 'Homo_sapiens_assembly38' | tr '\n' '|' | sed 's/|$//')" gs://gcp-public-data--broad-references/hg38/v0/ /data/genomes/`

### Configuration file
Create config file (`error_metrics.config`) of the following form: 

```
em_vc_demux_file=/home/ubuntu/proj/work/191015/420159_1p.demux.bam
em_vc_genome=/data/genomes/broad-references/hg38/v0/Homo_sapiens_assembly38.fasta
em_vc_output_dir=/home/ubuntu/proj/VariantCalling/work/191015/em
em_vc_number_to_sample=20000000 # Set -1 if the input file is already sampled
em_vc_number_of_cpus=12
```

Optionally, this could be a section in a general config file with header 
`[EM_VC]`

### Run
```
cd /home/ubuntu/proj/work/191015/
conda activate genomics.py3
python $HOME/software/VariantCalling/src/python/pipelines/error_rate_metrics_pipeline.py -c error_metrics.config
```

### Output
Text output files will be named: 
 - `.sort.metrics` - error metrics for unfiltered aligned data
 - `.sort.filter.metrics` - error metrics for filtered (>Q20) aligned data
 - `.idxstats` - alignment statistics

All outputs will be concatenated into HDF5 file `metrics.h5` with keys `bwa_alignment_stats` and `bwa_error_rates`

### Testing
On the machine `ec2-3-208-150-254.compute-1.amazonaws.com` run 
```
python $HOME/proj/VariantCalling/src/python/pipelines/error_metric_pipeline.py -c em.config 
```
in `/home/ec2-user/proj/VariantCalling/work/200112/`
