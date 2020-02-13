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

`gsutil -m rsync -x "$(gsutil ls gs://gcp-public-data--broad-references/hg38/v0/ | awk -F '/' '{print $6 }' | grep -v 'Homo_sapiens_assembly38' | tr '\n' '|' | sed 's/|$//')" gs://gcp-public-data--broad-references/hg38/v0/ /data/genomes/`

6. Copy helper files for variant calling
`aws s3 sync s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/ /data/genomes/broad-references/hg38/concordance/`

7. Copy additional files to the location of the genome: 

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/Homo_sapiens_assembly38.fasta.dict 
 /data/genomes/broad-references/hg38/v0/`

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/data/concordance/hg38/Homo_sapiens_assembly38.fasta.sizes 
/data/genomes/broad-references/hg38/v0/`

8. Download the latest gatk JAR, currently `gatk-package-ultima-v0.2-12-g4e6ad70-SNAPSHOT-local.jar`

`aws s3 cp s3://ultimagen-ilya-new/VariantCalling/jar/gatk-package-ultima-v0.2-12-g4e6ad70-SNAPSHOT-local.jar $HOME/software/gatk/`

9. Compile and install the recalibration code from `recalibration` repo. Assume into `$HOME/software/recalibration`


