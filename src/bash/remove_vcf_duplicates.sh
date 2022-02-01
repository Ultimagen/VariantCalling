INPUT=$1
if test -f $INPUT; then
  bcftools view -h $INPUT > $INPUT.nodup.vcf
  bcftools view -H $INPUT | sort -k 1,1 -k 2,2n | uniq >> $INPUT.nodup.vcf
  bcftools view -Oz  $INPUT.nodup.vcf > $INPUT.nodup.vcf.gz
  rm $INPUT.nodup.vcf
  tabix -p vcf $INPUT.nodup.vcf.gz;
fi