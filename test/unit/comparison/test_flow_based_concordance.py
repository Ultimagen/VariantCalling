import re
from os.path import join as pjoin

import pyfaidx

import ugvc.comparison.flow_based_concordance as fbc


def test_get_reference_from_region(tmpdir):
    fasta_seq = "A" * 50 + "NYQ" + "C" * 50
    fasta_file = pjoin(tmpdir, "test.fa")
    with open(fasta_file, "w") as out:
        out.write(">test\n")
        out.write(fasta_seq)
        out.write("\n")

    fai = pyfaidx.Fasta(pjoin(tmpdir, "test.fa"))
    refseq = fbc.get_reference_from_region(fai["test"], (1, len(fasta_seq) + 1))

    assert refseq.upper() == refseq
    assert len(refseq) == len(fasta_seq)
    assert re.search(r"^.ATGC", fasta_seq) is None
