#!bash
FILE=$1
bgzip -c $FILE > $FILE.gz
tabix -p vcf $FILE.gz
