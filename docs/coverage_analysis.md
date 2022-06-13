# coverage_analysis.py

This script collects coverage statistics of BAM/CRAM file over set of regions of interest.

```
usage: coverage_analysis [-h] [-i INPUT] [-o OUTPUT] -c COVERAGE_INTERVALS
                         [-r [REGION [REGION ...]]] [-w WINDOWS] [-q Q] [-Q Q]
                         [-l L] --reference REFERENCE
                         [--reference-gaps REFERENCE_GAPS]
                         [--centromeres CENTROMERES] [-j JOBS]
                         [--no_progress_bar]

Run full coverage analysis of an aligned bam/cram file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input bam or cram file
  -o OUTPUT, --output OUTPUT
                        Path to which output dataframe will be written, will
                        be created if it does not exist
  -c COVERAGE_INTERVALS, --coverage_intervals COVERAGE_INTERVALS
                        tsv file pointing to a dataframe detailing the various
                        intervals
  -r [REGION [REGION ...]], --region [REGION [REGION ...]]
                        Genomic region in samtools format - default is ['chr1'
                        'chr2' 'chr3' 'chr4' 'chr5' 'chr6' 'chr7' 'chr8'
                        'chr9' 'chr10' 'chr11' 'chr12' 'chr13' 'chr14' 'chr15'
                        'chr16' 'chr17' 'chr18' 'chr19' 'chr20' 'chr21'
                        'chr22' 'chrX' 'chrY' 'chrM']
  -w WINDOWS, --windows WINDOWS
                        Number of base pairs to bin coverage by (leave blank
                        for default [100, 1000, 10000, 100000])
  -q Q, -bq Q           Base quality theshold (default 0, samtools depth -q
                        parameter)
  -Q Q, -mapq Q         Mapping quality theshold (default 0, samtools depth -Q
                        parameter)
  -l L                  read length threshold (ignore reads shorter than
                        <int>) (default 0, samtools depth -l parameter)
  --reference REFERENCE
                        Reference fasta used for cram file compression, not
                        used for bam inputs
  --reference-gaps REFERENCE_GAPS
                        hg38 reference gaps, default taken from: hgdownload.cs
                        e.ucsc.edu/goldenpath/hg38/database/gap.txt.gz
  --centromeres CENTROMERES
                        centromeres file, extracted from: hgdownload.cse.ucsc.
                        edu/goldenpath/hg38/database/cytoBand.txt.gz
  -j JOBS, --jobs JOBS  Number of processes to run in parallel if INPUT is an
                        iterable (joblib convention - the number of CPUs)
  --no_progress_bar     Do not display progress bar for iterable INPUT
  ```
