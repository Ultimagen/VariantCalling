import os
from collections import Counter
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import numpy as np
import pyBigWig as pbw
import pyfaidx
import pytest

from ugvc.mrd.coverage_utils import _collect_coverage_per_motif

input_dir = get_resource_dir(__file__)
common_dir = pjoin(test_dir, "resources", "general")


@pytest.fixture
def data():
    seq = pyfaidx.Fasta(pjoin(common_dir, "sample.fasta"))["chr20"][:].seq.upper()
    return seq


@pytest.fixture(params=np.arange(50000, 100000, 10000))
def generate_coverage_test_set(tmpdir, request, data):
    start_pos = request.param

    with pbw.open(pjoin(tmpdir, "sample.bw"), "w") as out:
        out.addHeader([("chr20", 100000)])
        out.addEntries(["chr20"], [start_pos], ends=[start_pos + 100], values=[1.0])

    size = 5
    subsequences = [data[i - size - 1 : i + size] for i in range(start_pos, start_pos + 100)]
    subsequences = [x for x in subsequences if "N" not in x]
    cnt = Counter(subsequences)
    yield data, pjoin(tmpdir, "sample.bw"), cnt
    os.remove(pjoin(tmpdir, "sample.bw"))


def test_collect_coverage_per_motif(generate_coverage_test_set):
    sequence, coverage, expected = generate_coverage_test_set
    result = _collect_coverage_per_motif(sequence, coverage, 5, 1).set_index("motif_5")
    for k in result.index:
        assert result.loc[k, "count"] == expected[k], k + str(sequence.index(k))


def test_collect_coverage_per_motif_empty_dataset(data, tmpdir):
    with pbw.open(pjoin(tmpdir, "sample.bw"), "w") as out:
        out.addHeader([("chr20", 100000)])
        out.addEntries(["chr20"], [0], ends=[100], values=[0.0])

    result = _collect_coverage_per_motif(data, pjoin(tmpdir, "sample.bw"), 5, 1).set_index("motif_5")
    assert result.shape == (0, 1)
    assert "count" in result.columns
    os.remove(pjoin(tmpdir, "sample.bw"))


def test_collect_coverage_per_motif_long_chromosome(data, tmpdir):
    with pbw.open(pjoin(tmpdir, "sample.bw"), "w") as out:
        out.addHeader([("chr20", 10000000)])
        out.addEntries(["chr20"], [0], ends=[10000000], values=[1.0])
    result = _collect_coverage_per_motif(data, pjoin(tmpdir, "sample.bw"), 5, 10).set_index("motif_5")
    assert result["count"].sum() >= 990000  # Require less than 1M motifs since there are Ns on the end of the chrom
    os.remove(pjoin(tmpdir, "sample.bw"))
