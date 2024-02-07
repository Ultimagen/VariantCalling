import subprocess
import tempfile
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import numpy as np
import pandas as pd
import pysam
from pandas.testing import assert_frame_equal

from ugvc.mrd.featuremap_utils import FeaturemapAnnotator, FeatureMapFields
from ugvc.mrd.mrd_utils import (
    featuremap_to_dataframe,
    generate_synthetic_signatures,
    intersect_featuremap_with_signature,
    read_intersection_dataframes,
    read_signature,
)
from ugvc.vcfbed.variant_annotation import VcfAnnotator

general_inputs_dir = pjoin(test_dir, "resources", "general")
inputs_dir = get_resource_dir(__file__)
reference_fasta = pjoin(general_inputs_dir, "chr1_head", "Homo_sapiens_assembly38.fasta")
intersection_file_basename = "MRD_test_subsample.MRD_test_subsample_annotated_AF_vcf_gz_mrd_quality_snvs.intersection"


def test_featuremap_annotator(tmpdir):
    input_featuremap = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    tmpfile = f"{tmpdir}/output_featuremap.vcf.gz"
    VcfAnnotator.process_vcf(
        input_path=input_featuremap,
        output_path=tmpfile,
        annotators=[FeaturemapAnnotator()],
    )
    output_variants = pysam.VariantFile(tmpfile)
    forward_events = 0
    reverse_events = 0
    total_max_softclip_bases = 0
    for v in output_variants:
        assert FeatureMapFields.MAX_SOFTCLIP_LENGTH.value in v.info
        total_max_softclip_bases += int(v.info[FeatureMapFields.MAX_SOFTCLIP_LENGTH.value])
        if FeatureMapFields.MAX_SOFTCLIP_LENGTH.IS_FORWARD.value in v.info:
            forward_events += 1
        else:
            reverse_events += 1
    assert (
        FeatureMapFields.IS_DUPLICATE.value in output_variants.header.info
    ), f"{FeatureMapFields.IS_DUPLICATE.value} is not in info header {output_variants.header.info}"
    assert forward_events == 9
    assert reverse_events == 22
    assert total_max_softclip_bases == 81


def _assert_read_signature(signature, expected_signature, expected_columns=None, possibly_null_columns=None):
    expected_columns = expected_columns or [
        "ref",
        "alt",
        "id",
        "qual",
        "af",
    ]
    possibly_null_columns = possibly_null_columns or [
        "id",
        "qual",
    ]
    for c in expected_columns:
        assert c in signature.columns
        if c not in possibly_null_columns:
            assert not signature[c].isnull().all()
            assert (signature[c] == expected_signature[c]).all() or np.allclose(signature[c], expected_signature[c])


def test_read_signature_ug_mutect():
    signature = read_signature(pjoin(inputs_dir, "mutect_mrd_signature_test.vcf.gz"), return_dataframes=True)
    signature_no_sample_name = read_signature(
        pjoin(inputs_dir, "mutect_mrd_signature_test.no_sample_name.vcf.gz"),
        return_dataframes=True,
    )  # make sure we can read the dataframe even if the sample name could not be deduced from the header
    expected_signature = pd.read_hdf(pjoin(inputs_dir, "mutect_mrd_signature_test.expected_output.h5"))
    _assert_read_signature(
        signature,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
    )
    _assert_read_signature(
        signature_no_sample_name,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
        possibly_null_columns=["id", "qual", "depth_tumor_sample", "af"],
    )


def test_read_signature_ug_dv():
    signature = read_signature(pjoin(inputs_dir, "dv_mrd_signature_test.vcf.gz"), return_dataframes=True)
    expected_signature = pd.read_hdf(pjoin(inputs_dir, "dv_mrd_signature_test.expected_output.h5"))
    _assert_read_signature(
        signature,
        expected_signature,
        expected_columns=[
            "ref",
            "alt",
            "id",
            "qual",
            "af",
            "depth_tumor_sample",
            "cycle_skip_status",
            "gc_content",
            "left_motif",
            "right_motif",
            "mutation_type",
        ],
    )


def test_read_signature_external():
    signature = read_signature(pjoin(inputs_dir, "external_somatic_signature.vcf.gz"), return_dataframes=True)
    expected_signature = pd.read_hdf(pjoin(inputs_dir, "external_somatic_signature.expected_output.h5"))

    _assert_read_signature(signature, expected_signature)


def test_intersect_featuremap_with_signature(tmpdir):
    signature_file = pjoin(inputs_dir, "Pa_46.FreshFrozen.chr20.70039_70995.vcf.gz")
    featuremap_file = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    test_file = pjoin(inputs_dir, "intersected_featuremap.vcf.gz")

    output_intersection_file = pjoin(tmpdir, "intersected.vcf.gz")
    intersect_featuremap_with_signature(
        featuremap_file,
        signature_file,
        output_intersection_file=output_intersection_file,
    )
    cmd1 = f"bcftools view -H {output_intersection_file}"
    cmd2 = f"bcftools view -H {test_file}"
    assert subprocess.check_output(cmd1, shell=True) == subprocess.check_output(cmd2, shell=True)


def test_featuremap_to_dataframe():
    input_featuremap = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.vcf.gz")
    expected_featuremap_dataframe = pjoin(inputs_dir, "Pa_46.bsDNA.chr20_sample.parquet")
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.parquet")

        featuremap_dataframe = featuremap_to_dataframe(featuremap_vcf=input_featuremap, output_file=tmp_out_path)
        featuremap_dataframe_expected = pd.read_parquet(expected_featuremap_dataframe)
        _assert_read_signature(
            featuremap_dataframe,
            featuremap_dataframe_expected,
            expected_columns=[
                "chrom",
                "pos",
                "ref",
                "alt",
                "qual",
                "filter",
                "MLEAC",
                "MLEAF",
                "X_CIGAR",
                "X_EDIST",
                "X_FC1",
                "X_FC2",
                "X_FILTERED_COUNT",
                "X_FLAGS",
                "X_INDEX",
                "X_LENGTH",
                "X_MAPQ",
                "X_READ_COUNT",
                "X_RN",
                "X_SCORE",
                "X_SMQ_LEFT",
                "X_SMQ_LEFT_MEAN",
                "X_SMQ_RIGHT",
                "X_SMQ_RIGHT_MEAN",
                "a3",
                "ae",
                "as",
                "rq",
                "s2",
                "s3",
                "te",
                "tm",
                "ts",
            ],
            possibly_null_columns=["tm", "filter", "qual", "MLEAC", "MLEAF"],
        )


def test_read_intersection_dataframes():
    parsed_intersection_dataframe = read_intersection_dataframes(
        pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet"),
        return_dataframes=True,
    )
    parsed_intersection_dataframe_expected = pd.read_parquet(
        pjoin(inputs_dir, f"{intersection_file_basename}.parsed.expected_output.parquet")
    )
    parsed_intersection_dataframe2 = read_intersection_dataframes(
        [pjoin(inputs_dir, f"{intersection_file_basename}.expected_output.parquet")],
        return_dataframes=True,
    )
    assert_frame_equal(
        parsed_intersection_dataframe.reset_index(),
        parsed_intersection_dataframe_expected,
    )
    assert_frame_equal(
        parsed_intersection_dataframe2.reset_index(),
        parsed_intersection_dataframe_expected,
    )


def test_generate_synthetic_signatures(tmpdir):
    signature_file = pjoin(inputs_dir, "mutect_mrd_signature_test.vcf.gz")
    db_file = pjoin(
        inputs_dir,
        "pancan_pcawg_2020.mutations_hg38_GNOMAD_dbsnp_beds.sorted.Annotated.HMER_LEN.edited.chr19.vcf.gz",
    )
    synthetic_signature_list = generate_synthetic_signatures(
        signature_vcf=signature_file, db_vcf=db_file, n_synthetic_signatures=1, output_dir=tmpdir
    )
    signature = read_signature(synthetic_signature_list[0], return_dataframes=True)
    expected_signature = read_signature(pjoin(inputs_dir, "synthetic_signature_test.vcf.gz"), return_dataframes=True)
    # test that motif distribution is the same (0th order)
    assert (
        signature.groupby(["ref", "alt"]).value_counts() == expected_signature.groupby(["ref", "alt"]).value_counts()
    ).all()
