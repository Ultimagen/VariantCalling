from setuptools import find_packages, setup

packages = []
packages += find_packages(where="ugbio_utils/src/core", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/cnv", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/featuremap", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/mrd", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/ppmseq", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/srsnv", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/methylation", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/cloud_utils", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/filtering", exclude=["tests"])
packages += find_packages(where="ugbio_utils/src/comparison", exclude=["tests"])

setup(
    name="ugvc",
    version="0.27.0",
    packages=find_packages(exclude=["ugbio_utils*", "tests"]) + packages,
    package_dir={
        "ugbio_core": "ugbio_utils/src/core/ugbio_core",
        "ugbio_cnv": "ugbio_utils/src/cnv/ugbio_cnv",
        "ugbio_ppmseq": "ugbio_utils/src/ppmseq/ugbio_ppmseq",
        "ugbio_featuremap": "ugbio_utils/src/featuremap/ugbio_featuremap",
        "ugbio_srsnv": "ugbio_utils/src/srsnv/ugbio_srsnv",
        "ugbio_mrd": "ugbio_utils/src/mrd/ugbio_mrd",
        "ugbio_methylation": "ugbio_utils/src/methylation/ugbio_methylation",
        "ugbio_cloud_utils": "ugbio_utils/src/cloud_utils/ugbio_cloud_utils",
        "ugbio_filtering": "ugbio_utils/src/filtering/ugbio_filtering",
        "ugbio_comparison": "ugbio_utils/src/comparison/ugbio_comparison",
    },
    install_requires=[],
    scripts=[
        "ugvc/__main__.py",
        "ugbio_utils/src/comparison/ugbio_comparison/run_comparison_pipeline.py",
        "ugvc/pipelines/coverage_analysis.py",
        "ugbio_utils/src/core/ugbio_core/collect_existing_metrics.py",
        "ugbio_utils/src/core/ugbio_core/vcfbed/annotate_contig.py",
        "ugbio_utils/src/core/ugbio_core/sorter_stats_to_mean_coverage.py",
        "ugbio_utils/src/featuremap/ugbio_featuremap/featuremap_to_dataframe.py",
        "ugbio_utils/src/mrd/ugbio_mrd/intersect_featuremap_with_signature.py",
        "ugbio_utils/src/filtering/ugbio_filtering/training_prep_pipeline.py",
        "ugbio_utils/src/filtering/ugbio_filtering/train_models_pipeline.py",
        "ugbio_utils/src/filtering/ugbio_filtering/filter_variants_pipeline.py",
        "ugvc/pipelines/evaluate_concordance.py",
        "ugbio_utils/src/filtering/ugbio_filtering/sec/correct_systematic_errors.py",
        "ugbio_utils/src/filtering/ugbio_filtering/sec/sec_training.py",
        "ugvc/pipelines/vcfbed/gvcf_hcr.py",
        "ugvc/pipelines/denovo_recalibrated_qualities.py",
        "ugbio_utils/src/core/ugbio_core/convert_h5_to_json.py",
        "ugbio_utils/src/cnv/ugbio_cnv/filter_sample_cnvs.py",
        "ugbio_utils/src/cnv/ugbio_cnv/convert_cnv_results_to_vcf.py",
        "ugbio_utils/src/cnv/ugbio_cnv/plot_cnv_results.py",
        "ugbio_utils/src/cnv/ugbio_cnv/annotate_FREEC_segments.py",
        "ugbio_utils/src/filtering/ugbio_filtering/filter_variants_pipeline.py",
        "ugvc/pipelines/correct_genotypes_by_imputation.py",
        "ugbio_utils/src/cnv/ugbio_cnv/run_cnvpytor.py",
    ],
    entry_points={
        "console_scripts": [
            "run_no_gt_report = ugvc.pipelines.run_no_gt_report:run",
            "annotate_contig  = ugbio_core.vcfbed.annotate_contig:main",
            "collect_sv_stats = ugvc.pipelines.sv_stats_collect:main",
        ],
    },
    package_data={
        "ugvc": [
            "bash/run_ucsc_command.sh",
            "bash/remove_vcf_duplicates.sh",
            "bash/remove_empty_files.sh",
            "bash/index_vcf_file.sh",
            "bash/find_adapter_coords.sh",
        ],
        "": ["**/reports/*.ipynb", "**/reports/*.csv", "**/scripts/*.sh"],
    },
    install_package_data=True,
)
