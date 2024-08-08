import time

import pysam
from cyvcf2 import VCF
from profiling import create_annotation

vcf_in = "/data/Runs/call-FeatureMapProcess/input/Pa_46_333_LuNgs_08.vcf.gz"
# contig = "chr20"

create_annotation("start")
time.sleep(10)

#Cyvcf2 code
start_time = time.time()

for _ in range(10):
    records = 0
    for variant in VCF(vcf_in, threads=2):
        records += 1

print(f"cyvcf2 time: {time.time() - start_time}")
print(f"cyvcf2 records: {records}")
create_annotation("cyvcf2", time_start=start_time)

time.sleep(30)

#Pysam code
start_time = time.time()

for _ in range(10):
    records = 0

    with pysam.VariantFile(vcf_in) as input_variant_file:
        for record in input_variant_file:
            records += 1

print(f"pysam time: {time.time() - start_time}")
print(f"pysam records: {records}")
create_annotation("pysam", time_start=start_time)
