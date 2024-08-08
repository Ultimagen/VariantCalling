import sys
import time

import pysam
from cyvcf2 import VCF
from profiling import create_annotation

vcf_in = "/data/Runs/call-FeatureMapProcess/input/Pa_46_333_LuNgs_08.vcf.gz"
# contig = "chr20"


if "cyvcf2" in sys.argv:
    create_annotation("cyvcf2")
    start_time = time.time()

    for _ in range(10):
        records = 0
        for variant in VCF(vcf_in): #, threads=2):
            records += 1

    duration = time.time() - start_time
    print(f"cyvcf2 {duration=}")
    print(f"cyvcf2 {records=}")
    create_annotation(f"cyvcf2: {int(duration)}s", time_start=start_time)

    time.sleep(30)

if "pysam" in sys.argv:
    create_annotation("pysam")
    start_time = time.time()

    for _ in range(10):
        records = 0

        with pysam.VariantFile(vcf_in) as input_variant_file:
            for record in input_variant_file:
                records += 1

    duration = time.time() - start_time
    print(f"pysam {duration=}")
    print(f"pysam {records=}")
    create_annotation(f"pysam: {int(duration)}s", time_start=start_time)
