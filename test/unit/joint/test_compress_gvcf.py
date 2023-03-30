

import ugvc.joint.compress_gvcf as compress_gvcf
from os.path import join as pjoin
from test import get_resource_dir

def test_compress_gvcf(tmpdir):
    inputs_dir = get_resource_dir(__file__)
    input_file = pjoin(inputs_dir, "input.g.vcf.gz")
    output_file = pjoin(tmpdir, "compressed.g.vcf.gz")
    result = compress_gvcf.run(['compress_gvcf','--input_path',input_file,'--output_path',output_file])
    assert result == (4438, 1184)


def test_get_compressed_pl_into_3_values():

    result = compress_gvcf.get_compressed_pl_into_3_values((1,2,3))
    assert result == (1,2,3)

    result = compress_gvcf.get_compressed_pl_into_3_values((0,54,57,990, 2,990))
    assert result == (0,54,2)

    result = compress_gvcf.get_compressed_pl_into_3_values((0,54,57,990, 990,990,6,7,8,9))
    assert result == (0,6,7)