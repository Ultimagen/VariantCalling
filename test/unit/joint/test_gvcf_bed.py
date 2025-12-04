import filecmp
from test import get_resource_dir

import pybedtools
from ugbio_core.bed_utils import BedUtils

from ugvc.joint.gvcf_bed import gvcf_to_bed

inputs_dir = get_resource_dir(__file__)


def test_gvcf_to_bed_gt(tmpdir):
    input_file = f"{inputs_dir}/calls.g.vcf.gz"
    output_file = str(tmpdir.join("calls.observed.bed"))
    skipped = gvcf_to_bed(input_file, output_file, 20, gt=True)
    assert skipped == 241
    assert filecmp.cmp(output_file, f"{inputs_dir}/calls.expected.bed")
    output_file_mrg = str(tmpdir.join("calls.observed.mrg.bed"))
    pybedtools.BedTool(output_file).merge().saveas(output_file_mrg)
    assert BedUtils().count_bases_in_bed_file(output_file_mrg) == BedUtils().count_bases_in_bed_file(output_file)


def test_gvcf_to_bed_lt(tmpdir):
    input_file = f"{inputs_dir}/calls.g.vcf.gz"
    output_file = str(tmpdir.join("calls.observed.bed"))
    skipped = gvcf_to_bed(input_file, output_file, 20, gt=False)
    assert skipped == 241
    assert filecmp.cmp(output_file, f"{inputs_dir}/calls.expected.lt.bed")
    output_file_mrg = str(tmpdir.join("calls.observed.mrg.bed"))
    pybedtools.BedTool(output_file).merge(d=-1).saveas(output_file_mrg)
    assert BedUtils().count_bases_in_bed_file(output_file_mrg) == BedUtils().count_bases_in_bed_file(output_file)
