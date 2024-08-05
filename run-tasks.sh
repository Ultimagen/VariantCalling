#!/usr/bin/env bash

#time java -XX:GCTimeLimit=50 -XX:GCHeapFreeLimit=10 -Xms2g -jar ~/gatk-package-4.5.0.0-52-g2bf22ca-SNAPSHOT-local.jar FlowFeatureMapper -I /data/Runs/Pa_46.333_LuNgs_08.Lb_744.cram -O tmp1.vcf.gz -R "/data/Runs/Homo_sapiens_assembly38.fasta" --intervals /data/Runs/tmp/tmp1.interval_list --snv-identical-bases 5 --snv-identical-bases-after 5 --min-score 0 --limit-score 10 --read-filter MappingQualityReadFilter --minimum-mapping-quality 60 --include-dup-reads --flow-use-t0-tag --flow-fill-empty-bins-value 0.0001 --surrounding-median-quality-size 20 --copy-attr tm --copy-attr a3 --copy-attr rq --copy-attr as --copy-attr ts --copy-attr ae --copy-attr te --copy-attr s3 --copy-attr s2
#time java -XX:GCTimeLimit=50 -XX:GCHeapFreeLimit=10 -Xms2g -jar ~/gatk-package-4.5.0.0-52-g2bf22ca-SNAPSHOT-local.jar VariantFiltration -V tmp1.vcf.gz -O Pa_46_333_LuNgs_08.vcf.gz -R "/data/Runs/Homo_sapiens_assembly38.fasta" --intervals "/data/Runs/tmp/tmp1.interval_list"

# Flame graphs
# https://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html#Java

sudo bash
perf record -F 49 -a -g -- sleep 30; ./FlameGraph/jmaps
perf script > out.stacks01
cat out.stacks02 | ./FlameGraph/stackcollapse-perf.pl --all | grep -v cpu_idle | ./FlameGraph/flamegraph.pl --color=java --hash > out.stacks02.svg