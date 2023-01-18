from test import get_resource_dir

import ugvc.vcfbed.genotype as vcftools

inputs_dir = get_resource_dir(__file__)

def test_different_gt():
    assert not vcftools.different_gt('0|1', (0, 1))
    assert not vcftools.different_gt('1|0', (0, 1))
    assert vcftools.different_gt('1|1', (0, 1))
    assert vcftools.different_gt('1/2', (0, 1))
