from os.path import join as pjoin
from test import get_resource_dir

from ugvc.pipelines.vcfbed import intersect_bed_regions


def test_intersect_bed_regions(tmpdir):
    resource_dir = get_resource_dir(__file__)
    intersect_bed_regions.run(
        [
            "intersect_bed_regions",
            "--include-regions",
            pjoin(resource_dir, "arbitrary_region.chr20.bed"),
            pjoin(resource_dir, "ug_hcr.chr20.subset.bed"),
            "--exclude-regions",
            pjoin(resource_dir, "arbitrary_exclude_region.chr20.bed"),
            pjoin(resource_dir, "Homo_sapiens_assembly38.dbsnp138.chr20_subset.vcf.gz"),
            "--output-bed",
            pjoin(tmpdir, "output.bed"),
        ]
    )

    with open(pjoin(tmpdir, "output.bed"), "r") as f:
        result = f.readlines()
    with open(pjoin(resource_dir, "expected_output.bed"), "r") as f:
        expected_result = f.readlines()

    assert result == expected_result
