import os
import numpy as np
import pandas as pd
from os.path import join as pjoin, isfile
from tempfile import TemporaryDirectory
from pathmagic import PYTHON_TESTS_PATH
import pathmagic
import filecmp
from pathmagic import PYTHON_TESTS_PATH, COMMON
from python.pipelines.mrd_utils import create_control_signature


CLASS_PATH = "mrd_utils"


def test_coverage_analysis(tmpdir):
    reference_fasta = pjoin(PYTHON_TESTS_PATH, COMMON, "sample.fasta")
    signature = pjoin(
        PYTHON_TESTS_PATH,
        CLASS_PATH,
        "150382-BC04.filtered_signature.chr20_1_100000.vcf.gz",
    )
    control_signature = pjoin(
        PYTHON_TESTS_PATH,
        CLASS_PATH,
        "150382-BC04.filtered_signature.chr20_1_100000.control.vcf.gz",
    )
    output_control_signature = pjoin(tmpdir, "control_signature.vcf.gz")

    create_control_signature(
        signature,
        reference_fasta,
        control_signature_file_output=output_control_signature,
    )
    filecmp.cmp(control_signature, output_control_signature)
