from test import get_resource_dir

import ugvc.pipelines.vcfbed.gvcf_hcr_main as gvcf_hcr


def test_gvcf_hcr(tmpdir):
    inputs_dir = get_resource_dir(__file__)

    gvcf_hcr.run(
        [
            "--gvcf",
            f"{inputs_dir}/calls.g.vcf.gz",
            "--bed",
            f"{tmpdir}/calls.observed.bed",
            "--genome_file",
            "{tmpdir}/Homo_sapiens_assembly38.fasta.sizes",
        ]
    )
