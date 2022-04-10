INPUT=$1  # vcf/vcf.gz input file
OUTPUT=$2 # dedup .vcf.gz output file
if test -f $INPUT; then
  bcftools view -h $INPUT > $OUTPUT.vcf
  bcftools view -H $INPUT | sort -k 1,1 -k 2,2n | uniq >> $OUTPUT.vcf
  bcftools view -Oz  $OUTPUT.vcf > $OUTPUT
  rm $OUTPUT.vcf
  tabix -p vcf $OUTPUT;
fi
