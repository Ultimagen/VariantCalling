import os
from os.path import dirname
from test import get_resource_dir, test_dir

from ugvc.comparison.concordance_utils import read_hdf
from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.pipelines import run_comparison_pipeline


class TestRunComparisonPipeline:
    inputs_dir = get_resource_dir(__file__)
    general_inputs_dir = f"{test_dir}/resources/general/chr1_head"

    def test_run_comparison_pipeline(self, tmpdir):
        output_file = f"{tmpdir}/HG00239.vcf.gz"
        os.makedirs(dirname(output_file), exist_ok=True)

        run_comparison_pipeline.run(
            [
                "run_comparison_pipeline",
                "--n_parts",
                "0",
                "--hpol_filter_length_dist",
                "12",
                "10",
                "--input_prefix",
                f"{self.inputs_dir}/004797-UGAv3-51.filtered.chr1_1_1000000",
                "--output_file",
                f"{tmpdir}/004797-UGAv3-51.comp.h5",
                "--output_interval",
                f"{tmpdir}/004797-UGAv3-51.comp.bed",
                "--gtr_vcf",
                f"{self.inputs_dir}/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.chr1_1_1000000.vcf.gz",
                "--highconf_intervals",
                f"{self.inputs_dir}/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark_noinconsistent.chr1_1_1000000.bed",
                "--runs_intervals",
                f"{self.general_inputs_dir}/hg38_runs.conservative.bed",
                "--reference",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--reference_dict",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.dict",
                "--call_sample_name",
                "UGAv3-51",
                "--truth_sample_name",
                "HG004",
                "--ignore_filter_status",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--annotate_intervals",
                f"{self.general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/hmers_7_and_higher.bed",
                "--n_jobs",
                "4",
                "--coverage_bw_all_quality",
                f"{self.inputs_dir}/004797-UGAv3-51.chr1.q0.Q0.l0.w1.depth.chr1_1_1000000.bw",
                "--coverage_bw_high_quality",
                f"{self.inputs_dir}/004797-UGAv3-51.chr1.q0.Q20.l0.w1.depth.chr1_1_1000000.bw",
            ]
        )
        df = read_hdf(f"{tmpdir}/004797-UGAv3-51.comp.h5", key="chr1")
        assert {"tp": 346, "fn": 29, "fp": 27} == dict(df["classify"].value_counts())

    def test_run_comparison_pipeline_sentieon(self, tmpdir):
        output_file = f"{tmpdir}/004777-UGAv3-20.pred.chr1_1_1000000.comp.h5"
        os.makedirs(dirname(output_file), exist_ok=True)

        run_comparison_pipeline.run(
            [
                "run_comparison_pipeline",
                "--n_parts",
                "0",
                "--hpol_filter_length_dist",
                "12",
                "10",
                "--input_prefix",
                f"{self.inputs_dir}/004777-UGAv3-20.pred.chr1_1-1000000",
                "--output_file",
                f"{tmpdir}/004777-UGAv3-20.pred.chr1_1_1000000.comp.h5",
                "--output_interval",
                f"{tmpdir}/004777-UGAv3-20.comp.bed",
                "--gtr_vcf",
                f"{self.inputs_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.vcf.gz",
                "--highconf_intervals",
                f"{self.inputs_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.bed",
                "--runs_intervals",
                f"{self.general_inputs_dir}/hg38_runs.conservative.bed",
                "--reference",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--reference_dict",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.dict",
                "--call_sample_name",
                "UGAv3-20",
                "--truth_sample_name",
                "HG001",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--annotate_intervals",
                f"{self.general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/hmers_7_and_higher.bed",
                "--disable_reinterpretation",
                "--scoring_field",
                "ML_PROB",
                "--n_jobs",
                "2",
            ]
        )
        df = read_hdf(f"{tmpdir}/004777-UGAv3-20.pred.chr1_1_1000000.comp.h5", key="chr1")
        assert {"tp": 305, "fn": 11, "fp": 5} == dict(df["classify"].value_counts())

    def test_run_comparison_pipeline_dv(self, tmpdir):
        output_file = f"{tmpdir}/dv.pred.chr1_1-1000000.h5"
        os.makedirs(dirname(output_file), exist_ok=True)

        run_comparison_pipeline.run(
            [
                "run_comparison_pipeline",
                "--n_parts",
                "0",
                "--hpol_filter_length_dist",
                "12",
                "10",
                "--input_prefix",
                f"{self.inputs_dir}/dv.pred.chr1_1-1000000",
                "--output_file",
                f"{tmpdir}/dv.pred.chr1_1-1000000.h5",
                "--output_interval",
                f"{tmpdir}/dv.pred.chr1_1-1000000.bed",
                "--gtr_vcf",
                f"{self.inputs_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.vcf.gz",
                "--highconf_intervals",
                f"{self.inputs_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.bed",
                "--runs_intervals",
                f"{self.general_inputs_dir}/hg38_runs.conservative.bed",
                "--reference",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--reference_dict",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.dict",
                "--call_sample_name",
                "sm1",
                "--truth_sample_name",
                "HG001",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--annotate_intervals",
                f"{self.general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/hmers_7_and_higher.bed",
                "--disable_reinterpretation",
                "--scoring_field",
                "QUAL",
                "--revert_hom_ref",
                "--ignore_filter_status",
                "--n_jobs",
                "2",
            ]
        )
        df = read_hdf(f"{tmpdir}/dv.pred.chr1_1-1000000.h5", key="chr1")
        assert {"tp": 301, "fn": 15, "fp": 790} == dict(df["classify"].value_counts())
