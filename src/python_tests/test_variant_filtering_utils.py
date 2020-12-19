import pytest
import pathmagic
import pandas as pd
from python.pipelines import variant_filtering_utils


def test_blacklist_cg_insertions():
    rows = pd.DataFrame(
        {'alleles': [('C', 'T'),
                     ('CCG', 'C'),
                     ('G', 'GGC',)], 'filter': ['PASS', 'PASS', 'PASS']})
    rows = variant_filtering_utils.blacklist_cg_insertions(rows)
    filters = list(rows)
    assert filters == ['PASS', 'CG_NON_HMER_INDEL', 'CG_NON_HMER_INDEL']
