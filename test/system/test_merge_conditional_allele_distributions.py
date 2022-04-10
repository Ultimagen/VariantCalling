import glob
import os
from os.path import dirname
from test import get_resource_dir

from ugvc.pipelines.sec.merge_conditional_allele_distributions import (
    merge_conditional_allele_distributions,
)
from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions


class TestMergeConditionalAlleleDistributions:
    inputs_dir = get_resource_dir(__file__)

    def test_merge_conditional_allele_distributions(self, tmpdir):

        output_file = f"{tmpdir}/HG00239.vcf.gz"
        os.makedirs(dirname(output_file), exist_ok=True)
        cad_files_path = f"{tmpdir}/conditional_distribution_files.txt"
        output_prefix = f"{tmpdir}/conditional_allele_distributions"
        with open(cad_files_path, "w") as cad_files_fh:
            cad_files_fh.write(f"{self.inputs_dir}/HG00096.head.tsv\n")
            cad_files_fh.write(f"{self.inputs_dir}/HG00140.head.tsv\n")

        merge_conditional_allele_distributions(
            [
                "--conditional_allele_distribution_files",
                cad_files_path,
                "--output_prefix",
                output_prefix,
            ]
        )
        pickle_files = glob.glob(f"{output_prefix}.*.pkl")
        cads = ConditionalAlleleDistributions(pickle_files)

        cads_list_chr1 = [
            cad for cad in cads.distributions_per_chromosome["chr1"].values()
        ]
        cads_list_chr2 = [
            cad for cad in cads.distributions_per_chromosome["chr2"].values()
        ]
        cads_list_chr3 = [
            cad for cad in cads.distributions_per_chromosome["chr3"].values()
        ]

        # 87 common records (chr, pos)
        assert 119 == len(cads_list_chr1 + cads_list_chr2 + cads_list_chr3)

        # few records from chr2, and chr3 were read
        assert 7 == len(cads_list_chr2)
        assert 6 == len(cads_list_chr3)

        cad = cads.get_distributions_per_locus("chr1", 930192)
        assert {"T,TG"} == cad.get_possible_observed_alleles("0/0")
        assert (13, 55, 0) == cad.get_allele_counts("0/0", "T,TG", "T").get_counts()
        assert (42, 0, 0) == cad.get_allele_counts("0/0", "T,TG", "TG").get_counts()

        cad = cads.get_distributions_per_locus("chr1", 942741)
        assert {"CGG,C", "CG,C"} == cad.get_possible_observed_alleles("0/0")
        assert (5, 3, 0) == cad.get_allele_counts("0/0", "CGG,C", "CGG").get_counts()
        assert (5, 0, 0) == cad.get_allele_counts("0/0", "CGG,C", "C").get_counts()
        assert (1, 7, 0) == cad.get_allele_counts("0/0", "CG,C", "CG").get_counts()
        assert (1, 0, 0) == cad.get_allele_counts("0/0", "CG,C", "C").get_counts()
