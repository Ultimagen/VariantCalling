import os
from os.path import join as pjoin, isfile
from tempfile import TemporaryDirectory
from pathmagic import PYTHON_TESTS_PATH
from python.pipelines.coverage_analysis import run_full_coverage_analysis


f_in = pjoin(
    PYTHON_TESTS_PATH,
    "coverage_analysis",
    "170201-BC23.chr9_1000000_2001000.aligned.unsorted.duplicates_marked.bam",
)


def test_coverage_analysis():
    with TemporaryDirectory(prefix="/data/tmp" if os.path.isdir("/data/tmp") else "") as tmpdir:
        run_full_coverage_analysis(
            bam_file=f_in, out_path=tmpdir, regions=["chr9:1000000-2001000"], windows=[100_000]
        )
