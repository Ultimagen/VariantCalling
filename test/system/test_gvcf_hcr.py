import filecmp
from test import get_resource_dir

import pybedtools

import ugvc.pipelines.vcfbed.gvcf_hcr_main as gvcf_hcr
import ugvc.vcfbed.interval_file as interval_file


def test_gvcf_hcr(tmpdir):
    inputs_dir = get_resource_dir(__file__)
    gvcf_hcr.run(
        [
            "gvcf_hcr.py",
            "--gvcf",
            f"{inputs_dir}/calls.g.vcf.gz",
            "--bed",
            f"{tmpdir}/calls.observed.bed",
            "--genome_file",
            f"{inputs_dir}/chr9.fasta.fai",
            "--gq_threshold",
            "20",
            "--calling_region",
            f"{inputs_dir}/chr9.calling_regions.interval_list",
        ]
    )
    filecmp.cmp(f"{tmpdir}/calls.observed.bed", f"{inputs_dir}/hcr.expected.bed")
    bt = pybedtools.BedTool(f"{tmpdir}/calls.observed.bed")
    inf = interval_file.IntervalFile(interval=f"{inputs_dir}/chr9.calling_regions.interval_list")
    assert bt.subtract(a=f"{tmpdir}/calls.observed.bed", b=inf.as_bed_file()).count() == 0
