from __future__ import annotations

import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.comparison.variant_hit_fraction_caller import VariantHitFractionCaller
from ugbio_cloud_utils.cloud_sync import optional_cloud_sync
from ugbio_core.vcfbed.vcftools import index_vcf


# pylint: disable=too-many-instance-attributes
class QuickFingerprinter:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        sample_crams: dict[list[str]],
        ground_truth_vcfs: dict[str, str],
        hcrs: dict[str, str],
        ref: str,
        region: str,
        min_af_snps: int,
        min_af_germline_snps: int,
        min_hit_fraction_target: float,
        add_aws_auth_command: bool,
        out_dir: str,
        sp: SimplePipeline
    ):
        """
        Initialize the QuickFingerprinter with sample CRAMs, ground truth VCFs, HCRs, reference, region,
        filtering parameters, output directory, and pipeline object.
        Prepares ground truth files for comparison.
        """
        self.crams = sample_crams
        self.ground_truth_vcfs = ground_truth_vcfs
        self.hcrs = hcrs
        self.ref = ref
        self.region = region
        self.out_dir = out_dir
        self.min_af_snps = min_af_snps
        self.min_af_germline_snps = min_af_germline_snps
        self.min_hit_fraction_target = min_hit_fraction_target
        self.sp = sp
        self.add_aws_auth_command = add_aws_auth_command
        # Variant caller for hit fraction calculation
        self.vc = VariantHitFractionCaller(self.ref, self.out_dir, self.sp, self.min_af_snps, region)
        # Output file for results (now CSV)
        self.output_file = open(f"{self.out_dir}/quick_fingerprinting_results.csv", "w")

        os.makedirs(out_dir, exist_ok=True)

        # Prepare ground truth VCFs for each sample
        self.ground_truths_to_check = self.prepare_ground_truth()

    def prepare_ground_truth(self):
        """
        For each sample, prepare ground truth VCFs restricted to HCR and region.
        Returns a dict mapping sample_id to the processed ground truth VCF.
        """
        ground_truths_to_check = {}
        # Create a BED file for the region of interest
        self.sp.print_and_run(f"echo {self.region} | sed 's/:/\t/' | sed 's/-/\t/' > {self.out_dir}/region.bed")

        for sample_id in self.ground_truth_vcfs:
            # Sync ground truth VCF and HCR files from cloud if needed
            ground_truth_vcf = optional_cloud_sync(self.ground_truth_vcfs[sample_id], self.out_dir)
            hcr = optional_cloud_sync(self.hcrs[sample_id], self.out_dir)
            ground_truth_in_hcr = f"{self.out_dir}/{sample_id}_ground_truth_snps_in_hcr.vcf.gz"
            ground_truth_to_check_vcf = f"{self.out_dir}/{sample_id}_ground_truth_snps_to_check.vcf.gz"
            hcr_in_region = f"{self.out_dir}/{sample_id}_hcr_in_region.bed"

            # Intersect ground truth VCF with HCR, keep only SNPs
            self.sp.print_and_run(
                f"bedtools intersect -a {ground_truth_vcf} -b {hcr} -header | "
                + f"bcftools view --type snps -Oz -o  {ground_truth_in_hcr}"
            )
            index_vcf(self.sp, ground_truth_in_hcr)
            # Restrict to region of interest
            self.sp.print_and_run(
                f"bcftools view {ground_truth_in_hcr} -r {self.region} -Oz -o {ground_truth_to_check_vcf}"
            )
            index_vcf(self.sp, ground_truth_to_check_vcf)

            # Prepare HCR BED file restricted to region
            if self.region != "":
                self.sp.print_and_run(
                    f"bedtools intersect -a {hcr} -b {self.out_dir}/region.bed | "
                    f"sort -k 1,1 -k 2,2n > {hcr_in_region}"
                )
            else:
                self.sp.print_and_run(f"cp {hcr} {hcr_in_region}")

            ground_truths_to_check[sample_id] = ground_truth_to_check_vcf
        return ground_truths_to_check

    def print(self, msg: str):
        """
        Write a message to the output file.
        """
        self.output_file.write(msg + "\n")

    def check(self):
        """
        For each sample and each CRAM, call variants and compare to all ground truth VCFs.
        Print a CSV table with cram filename, sample_id, ground_truth_id, hit_fraction, and best_match.
        """
        # Print CSV header (without best_hit_fraction)
        self.print("cram_filename,sample_id,ground_truth_id,hit_fraction,best_match")

        for sample_id in self.crams:
            crams = self.crams[sample_id]
            for cram in crams:
                hit_fractions = []
                max_hit_fraction = 0
                best_match = None
                cram_base_name = os.path.basename(cram)
                called_vcf = f"{self.out_dir}/{cram_base_name}.calls.vcf.gz"
                local_bam = f"{self.out_dir}/{cram_base_name}.bam"
                # Convert CRAM to BAM for the region, with optional AWS auth
                if self.add_aws_auth_command:
                    self.sp.print_and_run(f"eval $(aws configure export-credentials --format env-no-export) samtools view {cram} -T {self.ref} {self.region} -b -o {local_bam}")
                else:
                    self.sp.print_and_run(f"samtools view {cram} -T {self.ref} {self.region} -b -o {local_bam}")

                # Call variants on the BAM
                self.vc.call_variants(local_bam, called_vcf, self.region, min_af=self.min_af_snps)

                # Calculate hit fractions for all ground truths
                hit_fraction_dict = {}
                for ground_truth_id, ground_truth_to_check_vcf in self.ground_truths_to_check.items():
                    hit_fraction, _, _ = self.vc.calc_hit_fraction(called_vcf, ground_truth_to_check_vcf)
                    hit_fraction_dict[ground_truth_id] = hit_fraction
                    if hit_fraction > max_hit_fraction:
                        max_hit_fraction = hit_fraction
                        best_match = ground_truth_id

                # Print a row for each ground_truth_id (without best_hit_fraction)
                for ground_truth_id, hit_fraction in hit_fraction_dict.items():
                    self.print(f"{cram_base_name},{sample_id},{ground_truth_id},{hit_fraction},{best_match}")

        self.output_file.close()
