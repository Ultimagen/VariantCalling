from test import get_resource_dir, test_dir
from ugvc.mrd.mrd_utils import annotate_featuremap, featuremap_to_dataframe
from os.path import join as pjoin
import pysam
from ugvc.vcfbed.variant_annotation import RefContextVcfAnnotator
from ugvc.mrd.mrd_utils import FeaturemapAnnotator
from ugvc.mrd.balanced_strand_utils import BalancedStrandVcfAnnotator, HistogramColumnNames


__general_inputs_dir = f"{test_dir}/resources/general/"

def test_annotate_featruremap(tmpdir):
    resource_dir = get_resource_dir(__file__)
    input_featuremap = pjoin(resource_dir, 'tp_featuremap_chr20.vcf.gz')
    output_featuremap = pjoin(tmpdir, 'tp_featuremap_chr20.annotated.vcf.gz')
    ref_fasta = pjoin(__general_inputs_dir, "sample.fasta")
    annotate_featuremap(input_featuremap, 
                        output_featuremap, 
                        ref_fasta=ref_fasta,
                        adapter_version='LA_v5and6', 
                        flow_order='TACG',
                        motif_length_to_annotate=3, 
                        max_hmer_length=20)
    out = pysam.VariantFile(output_featuremap)

    for info_field in ["X_CIGAR",
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
                       FeaturemapAnnotator.IS_FORWARD,
                       FeaturemapAnnotator.IS_DUPLICATE, 
                       FeaturemapAnnotator.MAX_SOFTCLIP_LENGTH, 
                       HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
                       HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
                       HistogramColumnNames.STRAND_RATIO_START.value,
                       HistogramColumnNames.STRAND_RATIO_END.value,
                       "trinuc_context",
                       "hmer_context_ref",
                       "hmer_context_alt",
                       "is_cycle_skip",
                       "prev_3bp",
                       "next_3bp"]:
        assert str(info_field) in out.header.info                                          

    

    