# noqa: W605  flake8: invalid escape sequence '\ ' - used by logo
# pylint: disable=anomalous-backslash-in-string, wrong-import-position

import sys
from os.path import dirname

from simppl import cli

# add package under repo-root to PYTHONPATH
path = f"{dirname(dirname(__file__))}"
if path not in sys.path:
    sys.path.insert(0, path)

from ugvc.cnv import filter_sample_cnvs
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
    evaluate_concordance,
    filter_variants_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
    vcfeval_flavors,
)
from ugvc.pipelines.lpr import filter_vcf_with_lib_prep_recalibration_model, train_lib_prep_recalibration_model
from ugvc.pipelines.mrd import (
    annotate_featuremap,
    balanced_strand_analysis,
    collect_coverage_per_motif,
    concat_dataframes,
    create_control_signature,
    create_hom_snv_featuremap,
    featuremap_to_dataframe,
    generate_synthetic_signatures,
    intersect_featuremap_with_signature,
    pileup_based_read_features,
    positional_error_rate_profile,
    prepare_data_from_mrd_pipeline,
    srsnv_inference,
    srsnv_training,
    substitution_error_rate,
)

# import pipeline modules implementing run(argv) method
from ugvc.pipelines.sec import assess_sec_concordance, correct_systematic_errors, sec_training, sec_validation
from ugvc.pipelines.vcfbed import annotate_contig, intersect_bed_regions
from ugvc.scripts import sorter_to_h5
from ugvc.somatic_cnv import bicseq2_post_processing
from ugvc.utils import cloud_sync

# create a list of imported pipeline modules
modules = [
    assess_sec_concordance,
    coverage_analysis,
    evaluate_concordance,
    filter_variants_pipeline,
    run_comparison_pipeline,
    train_models_pipeline,
    filter_sample_cnvs,
    convert_haploid_regions,
    correct_genotypes_by_imputation,
    vcfeval_flavors,
    bicseq2_post_processing,
]

sec_modules = [correct_systematic_errors, sec_training, sec_validation]

mrd_modules = [
    balanced_strand_analysis,
    substitution_error_rate,
    positional_error_rate_profile,
    collect_coverage_per_motif,
    concat_dataframes,
    create_control_signature,
    featuremap_to_dataframe,
    intersect_featuremap_with_signature,
    prepare_data_from_mrd_pipeline,
    pileup_based_read_features,
    generate_synthetic_signatures,
    annotate_featuremap,
    srsnv_training,
    srsnv_inference,
    create_hom_snv_featuremap,
]

methylation_modules = [
    concat_methyldackel_csvs,
    process_Mbias,
    process_mergeContext,
    process_mergeContextNoCpG,
    process_perRead,
]

joint_modules = [compress_gvcf]


misc_modules = [
    cloud_sync,
    sorter_to_h5,
]

lpr_modules = [
    train_lib_prep_recalibration_model,
    filter_vcf_with_lib_prep_recalibration_model,
]

vcfbed_modules = [annotate_contig, intersect_bed_regions]

modules.extend(mrd_modules)
modules.extend(sec_modules)
modules.extend(misc_modules)
modules.extend(methylation_modules)
modules.extend(joint_modules)
modules.extend(lpr_modules)
modules.extend(vcfbed_modules)


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
