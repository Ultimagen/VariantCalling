import pytest
from os.path import join as pjoin
from test import get_resource_dir

import ugvc.joint.cleanup_gvcf_before_calling as cgbc

def test_cleanup_gvcf_before_joint(tmpdir):
    inputs_dir = get_resource_dir(__file__)
    input_file = pjoin(inputs_dir, "subset.vcf.gz")
    output_file = pjoin(tmpdir, "output.vcf.gz")
    result = cgbc.filterOverlappingNoneGVCFs(input_file, output_file)
    assert result == (93620, 17)

