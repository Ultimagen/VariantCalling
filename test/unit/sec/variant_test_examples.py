import os

import pysam

from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.read_counts import ReadCounts


class TestSet:
    def __init__(self):
        self.vcf_header = pysam.VariantHeader()
        self.vcf_header.add_meta("contig", items=[("ID", "chr1")])
        self.vcf_header.add_meta("contig", items=[("ID", "chr9")])
        self.vcf_header.add_sample("HG001")
        self.vcf_header.add_meta(
            "FORMAT",
            items=[
                ("ID", "GT"),
                ("Number", 1),
                ("Type", "String"),
                ("Description", "Genotype"),
            ],
        )
        self.vcf_header.add_meta(
            "FORMAT",
            items=[
                ("ID", "SB"),
                ("Number", 4),
                ("Type", "Integer"),
                ("Description", "Strand bias vector"),
            ],
        )

        self.vcf = pysam.VariantFile("empty.vcf", "w", header=self.vcf_header)
        # cleanup file, which is not needed, it's just a side effect of initializing VariantFile
        os.remove("empty.vcf")


class NoVariantWithNoise(TestSet):
    def __init__(self):
        super().__init__()
        self.chrom = "chr1"
        self.pos = 50

        self.observed_variant = self.vcf.new_record(
            contig=self.chrom,
            start=self.pos,
            id="rs1",
            alleles=["C", "CG"],
            qual=53.29,
            filter="PASS",
            info={},
        )

        self.expected_distribution = ConditionalAlleleDistribution(
            conditioned_alleles="C",
            conditioned_genotype="0/0",
            observed_alleles="C,CG",
            allele_counts={"C": ReadCounts(12, 10), "CG": ReadCounts(0, 2)},
        )  # a bit of hmer noise


class KnownHetIns(TestSet):
    def __init__(self):
        super().__init__()
        self.chrom = "chr1"
        self.pos = 100

        self.observed_variant = self.vcf.new_record(
            contig=self.chrom,
            start=self.pos,
            id="rs1",
            alleles=["A", "AG"],
            qual=53.29,
            filter="PASS",
            info={},
        )

        self.expected_distribution = ConditionalAlleleDistribution(
            conditioned_alleles="A,AG",
            conditioned_genotype="0/0",
            observed_alleles="A,AG",
            allele_counts={"A": ReadCounts(12, 10), "AG": ReadCounts(0, 2)},
        )  # a bit of hmer noise
        self.expected_distribution.update_distribution(
            ConditionalAlleleDistribution(
                conditioned_alleles="A,AG",
                conditioned_genotype="0/1",
                observed_alleles="A,AG",
                allele_counts={"A": ReadCounts(7, 10), "AG": ReadCounts(10, 8)},
            )
        )


class UncorrelatedSnp(TestSet):
    def __init__(self):
        super().__init__()
        self.chrom = "chr1"
        self.pos = 300

        self.observed_variant = self.vcf.new_record(
            contig=self.chrom,
            start=self.pos,
            id="rs1",
            alleles=["A", "T"],
            qual=53.29,
            filter="PASS",
            info={},
        )

        self.expected_distribution = ConditionalAlleleDistribution(
            conditioned_alleles="A,T",
            conditioned_genotype="0/0",
            observed_alleles="A,T",
            allele_counts={"A": ReadCounts(120, 110), "T": ReadCounts(60, 50)},
        )  # mappability issues
        self.expected_distribution.update_distribution(
            ConditionalAlleleDistribution(
                conditioned_alleles="A,T",
                conditioned_genotype="0/1",
                observed_alleles="A,T",
                allele_counts={"A": ReadCounts(125, 112), "T": ReadCounts(57, 52)},
            )
        )


class NoReferenceGenotype(TestSet):
    def __init__(self):
        super().__init__()
        self.chrom = "chr9"
        self.pos = 39071

        self.observed_variant = self.vcf.new_record(
            contig=self.chrom,
            start=self.pos,
            id="rs1",
            alleles=["A", "G"],
            qual=53.29,
            filter="PASS",
            info={},
        )

        self.expected_distribution = ConditionalAlleleDistribution(
            conditioned_alleles="A,G",
            conditioned_genotype="0/1",
            observed_alleles="A,G",
            allele_counts={"A": ReadCounts(347, 339), "G": ReadCounts(614, 549)},
        )

        self.expected_distribution.update_distribution(
            ConditionalAlleleDistribution(
                conditioned_alleles="A,G",
                conditioned_genotype="1/1",
                observed_alleles="A,G",
                allele_counts={"A": ReadCounts(3, 4), "G": ReadCounts(224, 240)},
            )
        )

        self.expected_distribution.update_distribution(
            ConditionalAlleleDistribution(
                conditioned_alleles="A,G",
                conditioned_genotype="0/0",
                observed_alleles="A,G",
                allele_counts={},
                num_of_samples_with_observed_alleles=0,
            )
        )


class HomVarWithTwoEquivalentHetGenotypes(TestSet):
    def __init__(self):
        super().__init__()
        self.chrom = "chr1"
        self.pos = 400

        self.observed_variant = self.vcf.new_record(
            contig=self.chrom,
            start=self.pos,
            id="rs1",
            alleles=["A", "T"],
            qual=53.29,
            filter="PASS",
            info={},
        )

        self.expected_distribution = ConditionalAlleleDistribution(
            conditioned_alleles="A,T,C",
            conditioned_genotype="0/0",
            observed_alleles="A,T",
            allele_counts={"A": ReadCounts(120, 110), "T": ReadCounts(10, 20)},
        )  # mappability issues
        self.expected_distribution.update_distribution(
            ConditionalAlleleDistribution(
                conditioned_alleles="A,T,C",
                conditioned_genotype="0/2",
                observed_alleles="A,T",
                allele_counts={"A": ReadCounts(125, 112), "T": ReadCounts(10, 22)},
            )
        )
        self.expected_distribution.update_distribution(
            ConditionalAlleleDistribution(
                conditioned_alleles="A,T,C",
                conditioned_genotype="1/1",
                observed_alleles="A,T",
                allele_counts={"A": ReadCounts(0, 0), "T": ReadCounts(10, 22)},
            )
        )
