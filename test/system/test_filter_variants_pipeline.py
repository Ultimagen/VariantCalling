from test import get_resource_dir, test_dir

import ugvc.vcfbed.vcftools as vcftools
from ugvc.pipelines import filter_variants_pipeline


class TestFilterVariantPipeline:
    inputs_dir = get_resource_dir(__file__)
    general_inputs_dir = f"{test_dir}/resources/general/chr1_head"

    def test_filter_variants_pipeline(self, tmpdir):
        output_file = f"{tmpdir}/006919_no_frd_chr1_1_5000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/006919_no_frd_chr1_1_5000000.vcf.gz",
                "--model_file",
                f"{self.inputs_dir}/approximate_gt.model.pkl",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
                "--blacklist",
                f"{self.inputs_dir}/blacklist_example.chr1_1_1000000.pkl",
            ]
        )

        df = vcftools.get_vcf_df(output_file)
        assert {"LOW_SCORE": 2412, "PASS": 10266} == dict(df["filter"].value_counts())

    def test_filter_variants_pipeline_exact_model(self, tmpdir):
        output_file = f"{tmpdir}/036269-NA24143-Z0016.frd_chr1_1_5000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/036269-NA24143-Z0016.frd_chr1_1_5000000_unfiltered.vcf.gz",
                "--model_file",
                f"{self.inputs_dir}/exact_gt.model.pkl",
                "--output_file",
                output_file,
                "--custom_annotations",
                "LCR",
                "--custom_annotations",
                "MAP_UNIQUE",
                "--custom_annotations",
                "LONG_HMER",
                "--custom_annotations",
                "UG_HCR",
                "--ref_fasta",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--treat_multiallelics",
                "--recalibrate_genotype",
            ]
        )

        df = vcftools.get_vcf_df(output_file)
        assert {(0, 1): 5226, (0, 0): 5189, (1, 1): 2986, (1, 2): 63, (2, 2): 4} == dict(df["gt"].value_counts())
        assert {"PASS": 8317, "LOW_SCORE": 5151} == dict(df["filter"].value_counts())

    def test_filter_variants_pipeline_exact_model_no_gt_recal(self, tmpdir):
        output_file = f"{tmpdir}/036269-NA24143-Z0016.frd_chr1_1_5000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/036269-NA24143-Z0016.frd_chr1_1_5000000_unfiltered.vcf.gz",
                "--model_file",
                f"{self.inputs_dir}/exact_gt.model.pkl",
                "--output_file",
                output_file,
                "--custom_annotations",
                "LCR",
                "--custom_annotations",
                "MAP_UNIQUE",
                "--custom_annotations",
                "LONG_HMER",
                "--custom_annotations",
                "UG_HCR",
                "--ref_fasta",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--treat_multiallelics",
            ]
        )

        df = vcftools.get_vcf_df(output_file)
        assert {(0, 1): 10086, (1, 1): 3306, (1, 2): 76} == dict(df["gt"].value_counts())
        assert {"PASS": 8317, "LOW_SCORE": 5151} == dict(df["filter"].value_counts())

    def test_filter_variants_pipeline_blacklist_only(self, tmpdir):
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.blacklist_only.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
                "--blacklist",
                f"{self.inputs_dir}/blacklist_example.chr1_1_1000000.pkl",
            ]
        )

        df = vcftools.get_vcf_df(output_file)
        assert 4 == df[df["blacklst"].notna()]["blacklst"].count()

    def test_filter_variants_pipeline_cg_only(self, tmpdir):
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.blacklist_only.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
            ]
        )
        df = vcftools.get_vcf_df(output_file)
        assert 3 == df[df["blacklst"].notna()]["blacklst"].count()
