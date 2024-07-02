import os
from os.path import join as pjoin
from test import get_resource_dir

import pysam
import pytest

from ugvc.mrd.featuremap_consensus_utils import pileup_featuremap

inputs_dir = get_resource_dir(__file__)


@pytest.mark.parametrize(
    "genomic_interval, expected_num_variants",
    [("chr19:1-808924", 160), (None, 297)],
)
def test_pileup_featuremap(
    tmpdir,
    genomic_interval,
    expected_num_variants,
):
    # create input featuremap vcf file
    input_featuremap_vcf = pjoin(
        inputs_dir, "HG001_HG002_tumor_tumor_in_normal_31977.with_ml_qual.snps_in_hcr.chr19-1-1000000.vcf.gz"
    )

    # call the function with different arguments
    pileup_featuremap_vcf = pjoin(tmpdir, "pileup_featuremap.vcf.gz")
    pileup_featuremap(
        featuremap=input_featuremap_vcf,
        output_vcf=pileup_featuremap_vcf,
        genomic_interval=genomic_interval,
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(pileup_featuremap_vcf)
    # count the number of variants (excluding the header)
    num_variants = 0
    for _ in pysam.VariantFile(pileup_featuremap_vcf):
        num_variants += 1
    assert num_variants == expected_num_variants
