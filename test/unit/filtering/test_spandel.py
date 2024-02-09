import pathlib
import pickle
from test import get_resource_dir

import pyfaidx

import ugvc.filtering.spandel as spandel


def test_extract_allele_subset_from_multiallelic_spanning_deletion():
    inputs_dir = get_resource_dir(__file__)
    reference = pyfaidx.Fasta(str(pathlib.Path(inputs_dir, "ref_fragment.fa.gz")))
    spanning_deletion_examples_file = pathlib.Path(inputs_dir, "spanning_deletions.pkl")
    expected_results_file = pathlib.Path(inputs_dir, "expected_results_spanning_deletions.pkl")
    with open(expected_results_file, "rb") as f:
        expected_results = pickle.load(f)

    with open(spanning_deletion_examples_file, "rb") as f:
        spanning_deletions_examples, vcf_hdr_dct = pickle.load(f)
    results = [
        spandel.extract_allele_subset_from_multiallelic_spanning_deletion(
            example.iloc[1], example.iloc[0], (1, 2), vcf_hdr_dct, reference["chr21"]
        )
        for example in spanning_deletions_examples
    ]
    assert len(results) == len(expected_results)
    for i in range(len(results)):
        assert results[i].equals(expected_results[i])
