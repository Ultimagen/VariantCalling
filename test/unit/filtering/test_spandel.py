import pathlib
import pickle
from test import get_resource_dir

import pandas as pd
import pyfaidx
import pysam

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
        pd.testing.assert_series_equal(results[i], expected_results[i])


def test_split_multiallelic_variants_with_spandel():
    inputs_dir = get_resource_dir(__file__)
    reference = pyfaidx.Fasta(str(pathlib.Path(inputs_dir, "ref_fragment.fa.gz")))
    inputs_file = pathlib.Path(inputs_dir, "spanning_deletions.pkl")
    expected_result_file = pathlib.Path(inputs_dir, "expected_result_split_multiallelic.pkl")
    vcf_header = pysam.VariantFile(str(pathlib.Path(inputs_dir, "test_header.vcf.gz"))).header
    with open(inputs_file, "rb") as f:
        inputs, _ = pickle.load(f)

    with open(expected_result_file, "rb") as f:
        expected_results = pickle.load(f)

    results = [
        spandel.split_multiallelic_variants_with_spandel(
            input.iloc[1],
            input.iloc[0],
            vcf_header,
            reference["chr21"],
        )
        for input in inputs
    ]

    assert len(results) == len(expected_results)
    for i in range(len(results)):
        pd.testing.assert_frame_equal(results[i], expected_results[i])
