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
