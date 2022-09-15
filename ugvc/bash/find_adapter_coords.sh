#!/bin/bash

# Takes a bam file, converts to fastq, runs cutadapt, extracts coords, and adds to bam file
# also extracts one or two umi sequences
# Requires exactly 9 arguments in the specified order (this should be improved in the future by using getopt)
# There is no input checking yet either
# Function assumes that the sequence in the bam file does not have any Ns and that nucleotides are uppercase

set -e

if [ "$#" -ne 10 ]; then
  echo "Usage: $0 bam_infile conda_env left_adapter right_adapter left_umi_length right_umi_length error_rate_5p error_rate_3p min_overlap_5p min_overlap_3p" >&2
  exit 1
fi

bam_infile=$1
conda_env=$2
left_adapter=$3  # If empty string, signifies no 5p adapter
right_adapter=$4 # If empty string, signifies no 3p adapter
left_umi_len=$5 # If 0, do not extract umi at 5p
right_umi_len=$6 # If 0, do not extract umi at 3p
error_rate_5p=$7 # Should be between 0 and 1
error_rate_3p=$8 # Should be between 0 and 1
min_overlap_5p=$9 # An integer, minimal overlap
min_overlap_3p=${10} # An integer, minimal overlap

#ERROR_RATE_5p=0.15
#ERROR_RATE_3p=0.2
#MIN_OVERLAP_5p=5
#MIN_OVERLAP_3p=5

echo "bam_infile: $bam_infile"
echo "left_adapter: $left_adapter right_adapter: $right_adapter"
echo "left_umi_len: $left_umi_len right_umi_len: $right_umi_len"
echo "error_rate_5p: $error_rate_5p error_rate_3p: $error_rate_3p"
echo "min_overlap_5p: $min_overlap_5p min_overlap_3p: $min_overlap_3p"


# Converting to fastq
if [[ $bam_infile =~ ([^\/]*).bam$ ]]; then
        # Names of output files
        fastq_file="${BASH_REMATCH[1]}.fastq.gz"
        fastq_cutadapt_file="${BASH_REMATCH[1]}.cutadapt.fastq.gz"
        cutadapt_report_txt="${BASH_REMATCH[1]}.cutadapt_report.txt"
        cutadapt_report_json="${BASH_REMATCH[1]}.cutadapt.json"
        bam_outfile="${BASH_REMATCH[1]}.with_adapter_tags.bam"
    else
        echo "error in bam file name"
        exit 1
fi
echo "Converting to fastq $fastq_file..."
samtools fastq ${bam_infile} -0 ${fastq_file}

# Running cutadapt
# First checking which adapters to use
for_cutadapt=""
if [ ! -z "$left_adapter" ] && [ ! -z "$right_adapter" ]
then
     for_cutadapt="-a $left_adapter;max_error_rate=${error_rate_5p};min_overlap=${min_overlap_5p};required...$right_adapter;max_error_rate=${error_rate_3p};min_overlap=${min_overlap_3p}"
else
    if [ ! -z "$left_adapter" ]
    then
       for_cutadapt="-g $left_adapter;max_error_rate=${error_rate_5p};min_overlap=${min_overlap_5p}"
    fi
    if [ ! -z "$right_adapter" ]
    then
       for_cutadapt="-a $right_adapter;max_error_rate=${error_rate_3p};min_overlap=${min_overlap_3p}"
    fi
fi

echo "running cutadapt..."
orig_conda=$CONDA_DEFAULT_ENV
eval "$(conda shell.bash hook)"
set +e
( conda env list | grep -q $conda_env )
match=$?
set -e
if [ $match -ne 0 ]
  then
    >&2 echo "Please create environment $conda_env from src/setup/other_envs"
    exit 1
fi


conda activate $conda_env
cutadapt -j 0 ${for_cutadapt} --mask-adapter --json ${cutadapt_report_json} -o ${fastq_cutadapt_file} ${fastq_file}> ${cutadapt_report_txt}
conda deactivate
conda activate $orig_conda
echo "extracting coordinates and combining with bam file"
# The cat is for adding the header.  The header is concatenated with the paste, which pastes the bam and the cutadapt coordinates
cat <(samtools view -H ${bam_infile}) <(paste <(samtools view ${bam_infile})  <(zcat ${fastq_cutadapt_file} |
          awk 'BEGIN {OFS = "\t"}
              { if (NR%4 == 1) {
                  seq_id = substr($0,2)
                  }
                if (NR%4==2) {
                    print $0, length($0) #Preserving the length of the entire sequence
                }}' |
          sed 's/\([ACGT]\+\)/\t\1\t/' |  # Separating the masked adapters from the sequence
          awk -F "\t"  -v left_umi_len=${left_umi_len} -v right_umi_len=${right_umi_len} 'BEGIN {OFS = "\t";
                      for(i=1;i<=left_umi_len;i++) umi1_unknown = (umi1_unknown "N");
                      for(i=1;i<=right_umi_len;i++) umi2_unknown = (umi2_unknown "N")}
                  {
                   coord1=length($1)+1 # first coordinate starts with 1
                   if (NF == 2){ #In this case, the entire sequence was masked
                          coord1 = 0;
                          coord2 = 0;
                   } else {
                          coord2 = $4-length($3)+1
                   }
                   if  (left_umi_len > 0) { #Removing left umi
                       if (coord1 > 1) { #If a 5p adapter was found
                           coord1 = coord1 + left_umi_len
                           umi1 = substr($2,1,left_umi_len)
                       } else {
                           umi1 = umi1_unknown
                       }
                    }
                    if (right_umi_len > 0) { #Removing right umi
                        if (length($3) > 0) { # If the 3p adapter was found
                           coord2 = coord2 - right_umi_len
                           umi2_orig = substr($2,length($2)-right_umi_len+1,right_umi_len)
                           # revcom
                           umi2 = ""
                           split(umi2_orig, umi2_array, "")
                           for(i=right_umi_len;i!=0;i--) {
                               this_char = umi2_array[i]
                               if (this_char == "A") this_char="T"
                               else if (this_char == "T") this_char="A"
                               else if (this_char == "C") this_char="G"
                               else if (this_char == "G") this_char="C"
                               else if (this_char == "a") this_char="t"
                               else if (this_char == "t") this_char="a"
                               else if (this_char == "g") this_char="c"
                               else if (this_char == "c") this_char="g"
                               umi2=(umi2 this_char)
                            }
                       } else {
                           umi2 = umi2_unknown
                       }
                    }
                    if (left_umi_len == 0 && right_umi_len == 0) {
                        print "XF:i:"coord1,"XT:i:"coord2;
                    } else if (right_umi_len == 0) {
                        print "XF:i:"coord1, "XT:i:"coord2, "RX:Z:"umi1;
                    } else if (left_umi_len == 0) {
                        print "XF:i:"coord1, "XT:i:"coord2, "RX:Z:"umi2;
                    } else print "XF:i:"coord1, "XT:i:"coord2, "RX:Z:"umi1"-"umi2;}' )) | samtools view  -bo ${bam_outfile}
