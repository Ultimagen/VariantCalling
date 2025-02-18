import os
from os.path import dirname
from test import get_resource_dir
import pysam

from ugvc.pipelines import convert_haploid_regions
from ugvc.pipelines.convert_haploid_regions import load_regions_from_bed, in_regions
from ugbio_comparison.vcf_pipeline_utils import VcfPipelineUtils


class TestConvertHaploidRegions:
    inputs_dir = get_resource_dir(__file__)

    def test_convert_haploid_regions(self, tmpdir):
        input_file = f"{self.inputs_dir}/004777-X0024.annotated.AF_chr1_1_100000.vcf.gz"
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_100000.haploid.vcf.gz"
        haploid_regions_file = f"{self.inputs_dir}/haploid_regions.bed"
        os.makedirs(dirname(output_file), exist_ok=True)

        convert_haploid_regions.run(["prog_name",
                                     "--input_vcf", input_file,
                                     "--output_vcf", output_file,
                                     "--haploid_regions", haploid_regions_file])

        haploid_regions = load_regions_from_bed(haploid_regions_file)
        vpu = VcfPipelineUtils()
        vpu.index_vcf(output_file)
        vcf_reader = pysam.VariantFile(output_file)
        tp, fp, fn, tn = (0, 0, 0, 0)
        for variant in vcf_reader:
            genotype = variant.samples[0]['GT']
            if in_regions(variant.chrom, variant.pos, haploid_regions):

                if len(genotype) == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if len(genotype) == 2:
                    tn += 1
                else:
                    fp += 1

        assert tp == 16  # converted 16 variants to haploid, in haploid_regions
        assert tn == 93  # Left 93 variants as diploid
        assert fp == 0   # All haploid variants are in haploid_region
        assert fn == 0   # No diploid variants are left in haploid_region



