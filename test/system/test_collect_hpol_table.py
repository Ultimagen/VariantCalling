import os
from test import test_dir

import pandas as pd
import pyfaidx

import ugvc.scripts.collect_hpol_table as collect_hpol_table


def test_collect_hpol_table(tmpdir):
    general_inputs_dir = f"{test_dir}/resources/general/chr1_head"
    collect_hpol_table.run(
        [
            "collect_hpol_table.py",
            "--reference",
            f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta",
            "--collection_regions",
            f"{general_inputs_dir}/wgs_calling_regions.hg38.interval_list",
            "--output",
            f"{tmpdir}/hpol.table",
            "--max_hpol_length",
            "12",
            "--max_number_to_collect",
            "100",
        ]
    )
    assert os.path.exists(f"{tmpdir}/hpol.table")
    df = pd.read_csv(f"{tmpdir}/hpol.table", sep="\t", header=None, names=["chrom", "start", "hpol_length", "nuc"])
    fasta = pyfaidx.Fasta(f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta")
    # go over the rows of data frame, fetch the corresponding fasta sequence, check that the first nucleotide
    # is matching nuc
    for i in df.iterrows():
        seq = fasta[df.at[i[0], "chrom"]][df.at[i[0], "start"] : df.at[i[0], "start"] + df.at[i[0], "hpol_length"]].seq
        assert seq[0] == df.at[i[0], "nuc"]
        for j in range(df.at[i[0], "hpol_length"]):
            assert seq[0] == seq[j]
