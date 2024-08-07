from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pysam

from ugvc.mrd.featuremap_utils import FeatureMapFields
from ugvc.mrd.mrd_utils import annotate_featuremap
from ugvc.mrd.ppmSeq_utils import HistogramColumnNames

__general_inputs_dir = f"{test_dir}/resources/general/"


def test_annotate_featuremap(tmpdir):
    resource_dir = get_resource_dir(__file__)
    input_featuremap = pjoin(resource_dir, "tp_featuremap_chr20.vcf.gz")
    output_featuremap = pjoin(tmpdir, "tp_featuremap_chr20.annotated.vcf.gz")
    ref_fasta = pjoin(__general_inputs_dir, "sample.fasta")
    annotate_featuremap(
        input_featuremap,
        output_featuremap,
        ref_fasta=ref_fasta,
        ppmSeq_adapter_version="legacy_v5",
        flow_order="TGCA",
        motif_length_to_annotate=3,
        max_hmer_length=20,
    )
    out = pysam.VariantFile(output_featuremap)

    for info_field in [
        "X_CIGAR",
        "X_EDIST",
        "X_FC1",
        "X_FC2",
        "X_READ_COUNT",
        "X_FILTERED_COUNT",
        "X_FLAGS",
        "X_LENGTH",
        "X_MAPQ",
        "X_INDEX",
        "X_RN",
        "X_SCORE",
        "rq",
        FeatureMapFields.IS_FORWARD.value,
        FeatureMapFields.IS_DUPLICATE.value,
        FeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
        HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
        HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        HistogramColumnNames.STRAND_RATIO_START.value,
        HistogramColumnNames.STRAND_RATIO_END.value,
        "trinuc_context_with_alt",
        "hmer_context_ref",
        "hmer_context_alt",
        "is_cycle_skip",
        "prev_1",
        "prev_2",
        "prev_3",
        "next_1",
        "next_2",
        "next_3",
    ]:
        assert str(info_field) in out.header.info
