from pathmagic import PYTHON_TESTS_PATH
from os.path import join as pjoin
import python.pipelines.dev.sv_breakpoint_candidates as sv_breakpoint_candidates
from os.path import exists
import pandas as pd
from collections import Counter

CLASS_PATH = "sv_breakpoint_candidates"

def test_SV_breakpoint_candidates(tmp_path):
    csv_filename = pjoin(PYTHON_TESTS_PATH, CLASS_PATH, "150300-BC03.aligned.sorted.duplicates_marked.bam.SA.csv")
    output_file_basename = str(tmp_path / "temp_output")
    sv_breakpoint_candidates.SV_breakpoint_candidates(csv_filename, output_file_basename)

    assert exists(str(tmp_path / "temp_output.csv"))
    assert exists(str(tmp_path / "temp_output.bed"))

    csv = pd.read_csv(str(tmp_path / "temp_output.csv"), sep='\t')
    assert csv.shape[1] == 10
    assert len((csv['chr_id']).unique()) == 1
    assert min(csv['F_read_cnt']) > 5
    assert min(csv['R_read_cnt']) > 5

    bed = pd.read_csv(tmp_path / "temp_output.bed", sep='\t', header=None)
    assert bed.shape[1] == 4
    assert len((bed.iloc[:, 0]).unique()) == 1
    assert len((bed.iloc[:,0]).unique()) == 1
    assert all((bed.iloc[:,1]+10) == bed.iloc[:,2])





