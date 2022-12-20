from test import get_resource_dir, test_dir

import ugvc.vcfbed.vcftools as vcftools
from ugvc.pipelines import filter_variants_pipeline


class TestFilterVariantPipeline:
    inputs_dir = get_resource_dir(__file__)
    general_inputs_dir = f"{test_dir}/resources/general/chr1_head"

    def test_filter_variants_pipeline(self, tmpdir):
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz",
                "--model_file",
                f"{self.inputs_dir}/004777-X0024.model_rf_model_ignore_gt_incl_hpol_runs.pkl",
                "--model_name",
                "rf_model_ignore_gt_incl_hpol_runs",
                "--runs_file",
                f"{self.general_inputs_dir}/hg38_runs.conservative.bed",
                "--hpol_filter_length_dist",
                "12",
                "10",
                "--reference_file",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--blacklist_cg_insertions",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/hmers_7_and_higher.bed",
                "--output_file",
                output_file,
                "--blacklist",
                f"{self.inputs_dir}/blacklist_example.chr1_1_1000000.pkl",
            ]
        )

        df = vcftools.get_vcf_df(output_file)
        assert {"LOW_SCORE": 104, "PASS": 804} == dict(df["filter"].value_counts())

    def test_filter_variants_pipeline_blacklist_only(self, tmpdir):
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.blacklist_only.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz",
                "--runs_file",
                f"{self.general_inputs_dir}/hg38_runs.conservative.bed",
                "--reference_file",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
                "--blacklist",
                f"{self.inputs_dir}/blacklist_example.chr1_1_1000000.pkl",
            ]
        )

        df = vcftools.get_vcf_df(output_file)
        assert 4 == df[df['blacklst'].notna()]['blacklst'].count()

