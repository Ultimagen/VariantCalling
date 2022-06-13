import shutil
import subprocess
from os.path import basename, dirname, exists
from os.path import join as pjoin
from test import get_resource_dir

import pysam
import pytest

dn = dirname(__file__)
trimming_script = pjoin(dn[: dn.index("test")], "ugvc", "bash", "find_adapter_coords.sh")

test_params = [
    [
        "cutadapt",
        "^CTACACGACGCTCTTCCGATCT",
        "AGATCGGAAGAGCACACGTCTGAA",
        "0",
        "0",
        "0.15",
        "0.2",
        "10",
        "6",
    ],
    [
        "cutadapt",
        "^CTACACGACGCTCTTCCGATCT",
        "AGATCGGAAGAGCACACGTCTGAA",
        "8",
        "0",
        "0.15",
        "0.2",
        "10",
        "6",
    ],
    [
        "cutadapt",
        "^CTACACGACGCTCTTCCGATCT",
        "AGATCGGAAGAGCACACGTCTGAA",
        "8",
        "4",
        "0.15",
        "0.2",
        "10",
        "6",
    ],
    ["cutadapt", "^CTACACGACGCTCTTCCGATCT", "", "0", "0", "0.15", "0.2", "10", "6"],
    ["cutadapt", "", "AGATCGGAAGAGCACACGTCTGAA", "0", "0", "0.15", "0.2", "10", "6"],
]


expected_outputs = [pjoin(get_resource_dir(__file__), f"120461-BC23.test_output{x}.bam") for x in range(1, 6)]


@pytest.mark.parametrize("extra_params,expected", zip(test_params, expected_outputs))
def test_trimming_script(tmpdir, extra_params, expected):

    input_file = pjoin(get_resource_dir(__file__), "120461-BC23.for_test_input.bam")
    shutil.copyfile(input_file, pjoin(tmpdir, basename(input_file)))
    cmd = [trimming_script, pjoin(tmpdir, basename(input_file))] + extra_params
    subprocess.check_call(cmd, cwd=tmpdir)
    output_file = pjoin(tmpdir, basename(input_file).replace("bam", "with_adapter_tags.bam"))
    assert exists(output_file)
    assert _compare_bam_records(output_file, expected), f"{output_file} and {expected} are not identical"


def _compare_bam_records(bam1: str, bam2: str) -> bool:
    with pysam.AlignmentFile(bam1, check_sq=False) as input1, pysam.AlignmentFile(bam2, check_sq=False) as input2:
        for (rec1, rec2) in zip(input1, input2):
            if rec1.get_tag("XT") != rec2.get_tag("XT"):
                return False
            if rec2.get_tag("XF") != rec2.get_tag("XF"):
                return False
    return True
