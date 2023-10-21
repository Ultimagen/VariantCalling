import os
from os.path import join as pjoin
from test import get_resource_dir

import pysam
import pytest

from ugvc.mrd.featuremap_utils import filter_featuremap_with_bcftools_view

inputs_dir = get_resource_dir(__file__)


@pytest.mark.parametrize(
    "min_coverage, max_coverage, bcftools_include_filter, regions_string, expected_num_variants",
    [
        (None, None, None, "chr20\t85258\t85260\n", 79),
        (10, None, None, "chr20\t85258\t85260\n", 79),
        (None, 100, None, "chr20\t85258\t85260\n", 79),
        (None, None, "(X_SCORE >= 10)", "chr20\t85258\t85260\n", 79),
        (None, 100, "(X_SCORE >= 4) && (X_EDIST <= 5)", "chr20\t85258\t85260\n", 78),
        (None, 100, "(X_SCORE >= 4)", "chr20\t85258\t96322\n", 214),
        (84, 100, "(X_SCORE >= 4)", "chr20\t85258\t96322\n", 79),
    ],
)
def test_filter_featuremap_with_bcftools_view_with_params(
    tmpdir,
    min_coverage,
    max_coverage,
    bcftools_include_filter,
    regions_string,
    expected_num_variants,
):
    # create input featuremap vcf file
    input_featuremap_vcf = pjoin(inputs_dir, "333_LuNgs_08.annotated_featuremap.vcf.gz")

    # create regions file
    regions_file = pjoin(tmpdir, "regions.bed")
    with open(regions_file, "w") as f:
        f.write(regions_string)

    # call the function with different arguments
    intersect_featuremap_vcf = pjoin(tmpdir, "intersect_featuremap.vcf.gz")
    filter_featuremap_with_bcftools_view(
        input_featuremap_vcf=input_featuremap_vcf,
        intersect_featuremap_vcf=intersect_featuremap_vcf,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        regions_file=regions_file,
        bcftools_include_filter=bcftools_include_filter,
    )

    # check that the output file exists and has the expected content
    assert os.path.isfile(intersect_featuremap_vcf)
    # count the number of variants (excluding the header)
    num_variants = 0
    for _ in pysam.VariantFile(intersect_featuremap_vcf):
        num_variants += 1
    assert num_variants == expected_num_variants
