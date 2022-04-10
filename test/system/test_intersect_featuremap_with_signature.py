import filecmp
from os.path import join as pjoin
from test import get_resource_dir

from ugvc.pipelines.mrd.intersect_featuremap_with_signature import (
    intersect_featuremap_with_signature,
)


def test_intersect_featuremap_with_signature(tmpdir):
    input_dir = get_resource_dir(__file__)

    signature = pjoin(
        input_dir, "150382-BC04.filtered_signature.chr22_12693463.vcf.gz",
    )
    featuremap = pjoin(
        input_dir, "featuremap_150419-BC04.sorted.chr22_12693463.vcf.gz",
    )
    expected_intersection = pjoin(
        input_dir, "featuremap_150419-BC04.sorted.chr22_12693463.intersection.vcf.gz",
    )
    output_intersection = pjoin(tmpdir, "intersection.vcf.gz")
    intersect_featuremap_with_signature(
        featuremap_file=featuremap,
        signature_file=signature,
        output_intersection_file=output_intersection,
    )
    filecmp.cmp(output_intersection, expected_intersection)
