import filecmp
import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
from pandas.testing import assert_frame_equal

from ugvc.mrd.mrd_utils import (
    featuremap_to_dataframe,
    intersect_featuremap_with_signature,
    read_intersection_dataframes,
    read_signature,
)

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
reference_fasta = pjoin(general_inputs_dir, "chr1_head", "Homo_sapiens_assembly38.fasta")
intersection_file_basename = "MRD_test_subsample.MRD_test_subsample_annotated_AF_vcf_gz_mrd_quality_snvs.intersection"


def test_read_signature_ug_mutect():
    signature = read_signature(pjoin(inputs_dir, "mutect_mrd_signature_test.vcf.gz"))
    signature_no_sample_name = read_signature(pjoin(inputs_dir, "mutect_mrd_signature_test.no_sample_name.vcf.gz"))
    expected_output = pd.read_hdf(pjoin(inputs_dir, "mutect_mrd_signature_test.expected_output.h5"))

    assert_frame_equal(signature, expected_output)
    assert_frame_equal(
        signature_no_sample_name.drop(columns=["af", "depth_tumor_sample"]),
        expected_output.drop(columns=["af", "depth_tumor_sample"]),
    )  # make sure we can read the dataframe even if the sample name could not be deduced from the header


def test_read_signature_external():
    signature = read_signature(pjoin(inputs_dir, "external_somatic_signature.vcf.gz"))
    expected_output = pd.read_hdf(pjoin(inputs_dir, "external_somatic_signature.expected_output.h5"))

    assert_frame_equal(signature, expected_output)


def test_intersect_featuremap_with_signature():
    signature_file = pjoin(inputs_dir, "signature.chr19.vcf.gz")
    featuremap_file = pjoin(inputs_dir, "featuremap.chr19.vcf.gz")
    output_intersection_file = pjoin(inputs_dir, "featuremap.chr19.intersection.vcf.gz")
    output_intersection_file_headerless = pjoin(inputs_dir, "featuremap.chr19.intersection.NOHEADER.vcf")
    output_test_headerless = pjoin(inputs_dir, ".signature.matched.intersection.NOHEADER.vcf")
    intersect_featuremap_with_signature(
        featuremap_file, signature_file, output_intersection_file=output_intersection_file, is_matched=True
    )
    cmd1 = f"bcftools view -H {output_intersection_file} > {output_intersection_file_headerless}"
    subprocess.check_call(cmd1, shell=True)
    assert filecmp.cmp(output_intersection_file_headerless, output_test_headerless)


def test_featuremap_to_dataframe():
    featuremap_dataframe = featuremap_to_dataframe(
        pjoin(
            inputs_dir, "MRD_test_subsample.MRD_test_subsample_annotated_AF_vcf_gz_mrd_quality_snvs.intersection.vcf.gz"
        ),
        reference_fasta=reference_fasta,
    )
    featuremap_dataframe_expected = pd.read_parquet(
        pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet")
    )
    assert_frame_equal(featuremap_dataframe, featuremap_dataframe_expected)


def test_read_intersection_dataframes():
    parsed_intersection_dataframe = read_intersection_dataframes(
        pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet")
    )
    parsed_intersection_dataframe_expected = pd.read_parquet(
        pjoin(inputs_dir, f"{intersection_file_basename}.parsed.expected_output.parquet")
    )
    parsed_intersection_dataframe2 = read_intersection_dataframes(
        [pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet")]
    )
    assert_frame_equal(parsed_intersection_dataframe.reset_index(), parsed_intersection_dataframe_expected)
    assert_frame_equal(parsed_intersection_dataframe2.reset_index(), parsed_intersection_dataframe_expected)
