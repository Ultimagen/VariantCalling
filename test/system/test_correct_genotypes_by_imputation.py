import json
import os
import subprocess
from test import get_resource_dir, test_dir

import pandas as pd

from ugvc.pipelines import correct_genotypes_by_imputation


class TestCorrectGenotypesByImputation:
    inputs_dir = get_resource_dir(__file__)
    resources_dir = get_resource_dir(__file__)

    def test_correct_genotypes_by_imputation(self, tmpdir):
        os.makedirs(tmpdir, exist_ok=True)
        with open(os.path.join(tmpdir, "chrom_to_cohort_vcf.json"), "w") as f:
            json.dump(
                {
                    "chr1": os.path.join(self.resources_dir, "1kGP_phased_panel.chr1.vcf.gz"),
                    "chr2": os.path.join(self.resources_dir, "1kGP_phased_panel.chr2.vcf.gz"),
                },
                f,
            )
        with open(os.path.join(tmpdir, "chrom_to_plink.json"), "w") as f:
            json.dump(
                {
                    "chr1": os.path.join(self.resources_dir, "plink.chr1.GRCh38.chr.map"),
                    "chr2": os.path.join(self.resources_dir, "plink.chr2.GRCh38.chr.map"),
                },
                f,
            )
        input_vcf = os.path.join(self.resources_dir, "dv_013783-X0013.vcf.gz")
        output_vcf = os.path.join(tmpdir, "output.vcf.gz")
        correct_genotypes_by_imputation.run(
            [
                "correct_genotypes_by_imputation",
                "--input_vcf",
                input_vcf,
                "--chrom_to_cohort_vcf",
                os.path.join(tmpdir, "chrom_to_cohort_vcf.json"),
                "--chrom_to_plink",
                os.path.join(tmpdir, "chrom_to_plink.json"),
                "--output_vcf",
                output_vcf,
                "--temp_dir",
                str(tmpdir),
                "--epsilon",
                "0.01",
            ]
        )
        # Check that the numbers of variants which were changed are the same
        df0 = pd.read_csv(os.path.join(self.resources_dir, "out_counts.csv"), index_col=0)
        df1 = pd.read_csv(os.path.join(tmpdir, "output_counts.csv"), index_col=0)
        assert df1.equals(df0)
        # Check that the number of variants in the vcf did not change
        nvar_input = int(subprocess.check_output(f"bcftools view -H {input_vcf} | wc -l", shell=True))
        nvar_output = int(subprocess.check_output(f"bcftools view -H {output_vcf} | wc -l", shell=True))
        assert nvar_input == nvar_output
