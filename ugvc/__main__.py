# noqa: W605  flake8: invalid escape sequence '\ ' - used by logo
# pylint: disable=anomalous-backslash-in-string, wrong-import-position

import sys
from os.path import dirname

from simppl import cli

# add package under repo-root to PYTHONPATH
path = f"{dirname(dirname(__file__))}"
if path not in sys.path:
    sys.path.insert(0, path)

from ugbio_cloud_utils import cloud_sync

# import pipeline modules implementing run(argv) method
from ugbio_cnv import (
    annotate_FREEC_segments,
    bicseq2_post_processing,
    convert_cnv_results_to_vcf,
    filter_sample_cnvs,
    plot_cnv_results,
    plot_FREEC_neutral_AF,
)
from ugbio_comparison import run_comparison_pipeline
from ugbio_core import intersect_bed_regions, sorter_to_h5
from ugbio_core.vcfbed import annotate_contig
from ugbio_filtering import filter_variants_pipeline, train_models_pipeline, training_prep_pipeline
from ugbio_filtering.sec import assess_sec_concordance, correct_systematic_errors, sec_training, sec_validation
from ugbio_methylation import (
    concat_methyldackel_csvs,
    process_mbias,
    process_merge_context,
    process_merge_context_no_cp_g,
    process_per_read,
)

from ugvc.joint import compress_gvcf
from ugvc.pipelines import (
    convert_haploid_regions,
    correct_genotypes_by_imputation,
    coverage_analysis,
    denovo_recalibrated_qualities,
    evaluate_concordance,
    vcfeval_flavors,
)
from ugvc.pipelines.comparison import quick_fingerprinting
from ugvc.pipelines.deepvariant import training_set_consistency_check
from ugvc.pipelines.lpr import filter_vcf_with_lib_prep_recalibration_model, train_lib_prep_recalibration_model
from ugvc.pipelines.vcfbed import calibrate_bridging_snvs, gvcf_hcr

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
    annotate_FREEC_segments,
    plot_FREEC_neutral_AF,
]

sec_modules = [correct_systematic_errors, sec_training, sec_validation]

methylation_modules = [
    concat_methyldackel_csvs,
    process_mbias,
    process_merge_context,
    process_merge_context_no_cp_g,
    process_per_read,
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

vcfbed_modules = [annotate_contig, intersect_bed_regions, calibrate_bridging_snvs, gvcf_hcr]
deepvariant_modules = [training_set_consistency_check]
comparison_modules = [quick_fingerprinting]

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
