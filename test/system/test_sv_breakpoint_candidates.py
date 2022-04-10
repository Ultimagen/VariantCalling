from os.path import exists
from os.path import join as pjoin
from test import get_resource_dir

import pandas as pd

import ugvc.pipelines.dev.sv_breakpoint_candidates as sv_breakpoint_candidates

inputs_dir = get_resource_dir(__file__)


def test_SV_breakpoint_candidates(tmp_path):
    csv_filename = pjoin(
        inputs_dir, "150300-BC03.aligned.sorted.duplicates_marked.bam.SA.csv"
    )
    output_file_basename = str(tmp_path / "temp_output")
    sv_breakpoint_candidates.SV_breakpoint_candidates(
        csv_filename, output_file_basename
    )

    assert exists(str(tmp_path / "temp_output.csv"))
    assert exists(str(tmp_path / "temp_output.bed"))

    csv = pd.read_csv(str(tmp_path / "temp_output.csv"), sep="\t")
    assert csv.shape[1] == 10
    assert len((csv["chr_id"]).unique()) == 1
    assert min(csv["F_read_cnt"]) >= 5
    assert min(csv["R_read_cnt"]) >= 5

    bed = pd.read_csv(tmp_path / "temp_output.bed", sep="\t", header=None)
    assert bed.shape[1] == 4
    assert len((bed.iloc[:, 0]).unique()) == 1
    assert len((bed.iloc[:, 0]).unique()) == 1
    assert all((bed.iloc[:, 1] + 10) == bed.iloc[:, 2])


def test_SV_breakpoint_candidates_with_args(tmp_path):
    csv_filename = pjoin(
        inputs_dir, "150300-BC03.aligned.sorted.duplicates_marked.bam.SA.csv"
    )
    output_file_basename = str(tmp_path / "temp_output")
    sv_breakpoint_candidates.SV_breakpoint_candidates(
        csv_filename, output_file_basename, minappearances=4
    )

    assert exists(str(tmp_path / "temp_output.csv"))
    assert exists(str(tmp_path / "temp_output.bed"))

    csv = pd.read_csv(str(tmp_path / "temp_output.csv"), sep="\t")
    assert csv.shape[1] == 10
    assert len((csv["chr_id"]).unique()) == 1
    assert min(csv["F_read_cnt"]) >= 4
    assert min(csv["R_read_cnt"]) >= 4

    bed = pd.read_csv(tmp_path / "temp_output.bed", sep="\t", header=None)
    assert bed.shape[1] == 4
    assert len((bed.iloc[:, 0]).unique()) == 1
    assert len((bed.iloc[:, 0]).unique()) == 1
    assert all((bed.iloc[:, 1] + 10) == bed.iloc[:, 2])
