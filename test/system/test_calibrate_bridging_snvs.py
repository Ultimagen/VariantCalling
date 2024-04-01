from test import get_resource_dir, test_dir

import pysam
import pytest

from ugvc.pipelines.vcfbed import calibrate_bridging_snvs

__inputs_dir = get_resource_dir(__file__)
__general_inputs_dir = f"{test_dir}/resources/general/chr1_head"


def count_pass(variant_file) -> int:
    pass_variants = 0
    for variant in pysam.VariantFile(variant_file):
        if "PASS" in variant.filter:
            pass_variants += 1
    return pass_variants


def count_qual_over(variant_file, min_qual) -> int:
    pass_variants = 0
    for variant in pysam.VariantFile(variant_file):
        if variant.qual >= min_qual:
            pass_variants += 1
    return pass_variants


@pytest.mark.parametrize(
    "min_query_hmer_size, expected_pass, expected_qual20", [["2", 95, 64], ["3", 93, 62], ["4", 92, 61], ["5", 92, 61]]
)
def test_bridging_snvs_calibration(tmpdir, min_query_hmer_size, expected_pass, expected_qual20):
    __output = f"{tmpdir}/COLO-829_031865_v1.2.annotated.filt.chr1_head.fix_bridging_snvs.vcf.gz"
    __input = f"{__inputs_dir}/COLO-829_031865_v1.2.annotated.filt.chr1_head.vcf.gz"
    calibrate_bridging_snvs.run(
        [
            "calibrate_bridging_snvs",
            "--vcf",
            __input,
            "--reference",
            f"{__general_inputs_dir}/Homo_sapiens_assembly38.fasta",
            "--output",
            __output,
            "--min_query_hmer_size",
            min_query_hmer_size,
        ]
    )

    pass_variants_input = count_pass(__input)
    pass_variants_output = count_pass(__output)
    qual20_input = count_qual_over(__input, 20)
    qual20_output = count_qual_over(__output, 20)
    assert pass_variants_input == 91
    assert qual20_input == 60
    assert pass_variants_output == expected_pass
    assert qual20_output == expected_qual20
