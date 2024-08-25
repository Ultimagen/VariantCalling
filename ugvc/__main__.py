# noqa: W605  flake8: invalid escape sequence '\ ' - used by logo
# pylint: disable=anomalous-backslash-in-string, wrong-import-position

import sys
from os.path import dirname

from simppl import cli

# add package under repo-root to PYTHONPATH
path = f"{dirname(dirname(__file__))}"
if path not in sys.path:
    sys.path.insert(0, path)

# import pipeline modules implementing run(argv) method
from ugbio_cnv import convert_cnv_results_to_vcf, filter_sample_cnvs, plot_cnv_results , bicseq2_post_processing , annotate_FREEC_segments
from ugvc.joint import compress_gvcf
from ugvc.methylation import (
    concat_methyldackel_csvs,
    process_Mbias,
    process_mergeContext,
    process_mergeContextNoCpG,
    process_perRead,
)
from ugvc.pipelines import (
    convert_haploid_regions,
    correct_genotypes_by_imputation,
    coverage_analysis,
    denovo_recalibrated_qualities,
    evaluate_concordance,
    filter_variants_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
    training_prep_pipeline,
    vcfeval_flavors,
)
from ugvc.pipelines.comparison import quick_fingerprinting
from ugvc.pipelines.deepvariant import training_set_consistency_check
from ugvc.pipelines.lpr import filter_vcf_with_lib_prep_recalibration_model, train_lib_prep_recalibration_model
from ugbio_featuremap import featuremap_utils
from ugbio_featuremap.pipelines import annotate_featuremap, create_hom_snv_featuremap, featuremap_to_dataframe, \
    pileup_featuremap, sorter_stats_to_mean_coverage
from ugbio_mrd.pipelines import (
    generate_synthetic_signatures,
    intersect_featuremap_with_signature,
    prepare_data_from_mrd_pipeline
)
from ugbio_ppmseq.pipelines import (
    ppmSeq_qc_analysis
)
from ugbio_srsnv.pipelines import (
    srsnv_inference,
    srsnv_training
)
from ugvc.pipelines.sec import assess_sec_concordance, correct_systematic_errors, sec_training, sec_validation
from ugvc.pipelines.vcfbed import annotate_contig, calibrate_bridging_snvs, intersect_bed_regions
from ugvc.scripts import sorter_to_h5
from ugvc.utils import cloud_sync

# create a list of imported pipeline modules
modules = [
    assess_sec_concordance,
    coverage_analysis,
    evaluate_concordance,
    filter_variants_pipeline,
    training_prep_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
    filter_sample_cnvs,
    convert_cnv_results_to_vcf,
    plot_cnv_results,
    convert_haploid_regions,
    correct_genotypes_by_imputation,
    vcfeval_flavors,
    bicseq2_post_processing,
    annotate_FREEC_segments
]

sec_modules = [correct_systematic_errors, sec_training, sec_validation]

mrd_modules = [
    ppmSeq_qc_analysis,
    sorter_stats_to_mean_coverage,
    featuremap_to_dataframe,
    intersect_featuremap_with_signature,
    prepare_data_from_mrd_pipeline,
    generate_synthetic_signatures,
    annotate_featuremap,
    srsnv_training,
    srsnv_inference,
    create_hom_snv_featuremap,
    pileup_featuremap,
]

methylation_modules = [
    concat_methyldackel_csvs,
    process_Mbias,
    process_mergeContext,
    process_mergeContextNoCpG,
    process_perRead,
]

joint_modules = [compress_gvcf, denovo_recalibrated_qualities]


misc_modules = [
    cloud_sync,
    sorter_to_h5,
]

lpr_modules = [
    train_lib_prep_recalibration_model,
    filter_vcf_with_lib_prep_recalibration_model,
]

vcfbed_modules = [annotate_contig, intersect_bed_regions, calibrate_bridging_snvs]
deepvariant_modules = [training_set_consistency_check]
comparison_modules = [quick_fingerprinting]

modules.extend(mrd_modules)
modules.extend(sec_modules)
modules.extend(misc_modules)
modules.extend(methylation_modules)
modules.extend(joint_modules)
modules.extend(lpr_modules)
modules.extend(vcfbed_modules)
modules.extend(deepvariant_modules)
modules.extend(comparison_modules)

LOGO = """
      __    __    ___________    ____  ______
    |  |  |  |  /  _____\   \  /   / /      | # noqa: W605
    |  |  |  | |  |  __  \   \/   / |  ,----' # noqa: W605
    |  |  |  | |  | |_ |  \      /  |  |      # noqa: W605
    |  `--'  | |  |__| |   \    /   |  `----. # noqa: W605
     \______/   \______|    \__/     \______| # noqa: W605

    Ultima genomics variant calling toolkit
    """

# initialize cli
ugvc_cli = cli.CommandLineInterface(__file__, LOGO, modules)
ugvc_cli.run(sys.argv)
