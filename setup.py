from setuptools import setup, find_packages

setup(
    name="ugvc",
    version="0.7",
    packages=find_packages(),
    install_requires=[],
    scripts=[
        "ugvc/__main__.py",
        "ugvc/pipelines/run_comparison_pipeline.py",
        "ugvc/pipelines/coverage_analysis.py",
        "ugvc/pipelines/collect_existing_metrics.py",
        "ugvc/pipelines/mrd/snp_error_rate.py",
        "ugvc/pipelines/mrd/positional_error_rate_profile.py",
        "ugvc/pipelines/mrd/collect_coverage_per_motif.py",
        "ugvc/pipelines/mrd/concat_dataframes.py",
        "ugvc/pipelines/mrd/create_control_signature.py",
        "ugvc/pipelines/mrd/featuremap_to_dataframe.py",
        "ugvc/pipelines/mrd/intersect_featuremap_with_signature.py",
        "ugvc/pipelines/mrd/prepare_data_from_mrd_pipeline.py",
        "ugvc/pipelines/train_models_pipeline.py",
        "ugvc/pipelines/filter_variants_pipeline.py",
        "ugvc/pipelines/evaluate_concordance.py",
        "ugvc/pipelines/run_no_gt_report.py",
        "ugvc/pipelines/sec/correct_systematic_errors.py",
        "ugvc/pipelines/sec/sec_training.py",
        "ugvc/scripts/convert_h5_to_json.py"
    ],
    package_data={'ugvc': [
        "bash/run_ucsc_command.sh",
        "bash/remove_vcf_duplicates.sh",
        "bash/remove_empty_files.sh",
        "bash/index_vcf_file.sh"]},
    install_package_data=True
)