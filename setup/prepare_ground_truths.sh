#!/bin/bash

set -e

input_dir=$1
ref_dict=$2

if [ -z $input_dir ] || [ -z $ref_dict ]; then
  echo "Missing arguments"
  exit 1
fi

broad_header='broad-header'
vcf_files_list=$(ls $input_dir/*.vcf.gz | grep -v "$broad_header")
echo $vcf_files_list

if [ ${#vcf_files_list[@]} -eq 0 ]; then
  echo "No .vcf.gz files in $input_dir without $broad_header"
  exit 0
fi

for item in $vcf_files_list; do
  echo "Processing $item"
  item_basename=$(echo $item | sed 's/.vcf.gz//')
  item_target_name="$item_basename.$broad_header.vcf"
  picard UpdateVcfSequenceDictionary I=$item \
                                     O=$item_target_name \
                                     SD=$ref_dict

  bgzip_cmd="bgzip -f $item_target_name"
  echo "$bgzip_cmd"
  $bgzip_cmd

  bcftools_cmd="bcftools index -t $item_target_name.gz"
  echo "$bcftools_cmd"
  $bcftools_cmd
done

rm_cmd="rm -f $vcf_files_list"
echo "$rm_cmd"
$rm_cmd

vcf_index_files_list=$(ls $input_dir/*.vcf.gz.tbi | grep -v "$broad_header")
rm_idx_cmd="rm -f $vcf_index_files_list"
$rm_idx_cmd

json="{"
for i in $(seq $(ls $input_dir/HG[0-9]* | tail -1 | sed 's/\(.*HG\)\([0-9]\)*\(.*\)/\2/')); do
  if [[ $i -lt 10 ]]; then
    num="00$i"
  elif [[ $i -lt 100 ]]; then
    num="0$i"
  else
    num="0$i"
  fi
  list="$(ls $input_dir/HG$num* | sed 's/\([^ ;]*\)/"\1",/g')"
  json+="\"HG$num\":[${list:0:-1}],"
done

json="${json:0:-1}}"
echo $json

if jq -e . >/dev/null 2>&1 <<<"$json"; then
    echo "JSON is valid"
    echo $json | jq > $input_dir/ground_truths.json
else
    echo "Failed to parse JSON, or got false/null"
    exit 1
fi
