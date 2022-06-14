from os.path import join as pjoin
from test import get_resource_dir, test_dir

import numpy as np
import pandas as pd

from ugvc.pipelines.coverage_analysis import run_full_coverage_analysis

inputs_dir = get_resource_dir(__file__)
general_inputs_dir = f"{test_dir}/resources/general/"

f_in = pjoin(
    inputs_dir,
    "170201-BC23.chr9_1000000_2001000.aligned.unsorted.duplicates_marked.bam",
)
f_ref = pjoin(
    inputs_dir,
    "170201-BC23.chr9_1000000_2001000.coverage_percentiles.parquet",
)


def test_coverage_analysis(tmpdir):
    run_full_coverage_analysis(
        bam_file=f_in,
        out_path=tmpdir,
        regions=["chr9:1000000-2001000"],
        windows=[100_000],
        ref_fasta=pjoin(general_inputs_dir, "sample.fasta"),
        coverage_intervals_dict=pjoin(inputs_dir, "coverage_chr9_extended_intervals.tsv"),
    )
    df = pd.read_hdf(pjoin(tmpdir, "170201-BC23.coverage_stats.q0.Q0.l0.h5"), "percentiles")
    df_ref = pd.read_parquet(f_ref)
    assert np.allclose(df.fillna(-1), df_ref.fillna(-1))
