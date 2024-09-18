from __future__ import annotations

import os

from simppl.simple_pipeline import SimplePipeline

from ugvc.comparison.variant_hit_fraction_caller import VariantHitFractionCaller
from ugvc.utils.cloud_sync import optional_cloud_sync
from ugvc.vcfbed.vcftools import index_vcf


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
        out_dir: str,
        sp: SimplePipeline,
    ):
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
        self.vc = VariantHitFractionCaller(self.ref, self.out_dir, self.sp, self.min_af_snps, region)

        os.makedirs(out_dir, exist_ok=True)

        self.ground_truths_to_check = self.prepare_ground_truth()

    def prepare_ground_truth(self):
        ground_truths_to_check = {}
        self.sp.print_and_run(f"echo {self.region} | sed 's/:/\t/' | sed 's/-/\t/' > {self.out_dir}/region.bed")

        for sample_id in self.ground_truth_vcfs:
            ground_truth_vcf = optional_cloud_sync(self.ground_truth_vcfs[sample_id], self.out_dir)
            hcr = optional_cloud_sync(self.hcrs[sample_id], self.out_dir)
            ground_truth_in_hcr = f"{self.out_dir}/{sample_id}_ground_truth_snps_in_hcr.vcf.gz"
            ground_truth_to_check_vcf = f"{self.out_dir}/{sample_id}_ground_truth_snps_to_check.vcf.gz"
            hcr_in_region = f"{self.out_dir}/{sample_id}_hcr_in_region.bed"

            self.sp.print_and_run(
                f"bedtools intersect -a {ground_truth_vcf} -b {hcr} -header | "
                + f"bcftools view --type snps -Oz -o  {ground_truth_in_hcr}"
            )
            index_vcf(self.sp, ground_truth_in_hcr)
            self.sp.print_and_run(
                f"bcftools view {ground_truth_in_hcr} -r {self.region} -Oz -o {ground_truth_to_check_vcf}"
            )
            index_vcf(self.sp, ground_truth_to_check_vcf)

            if self.region != "":
                self.sp.print_and_run(
                    f"bedtools intersect -a {hcr} -b {self.out_dir}/region.bed | "
                    f"sort -k 1,1 -k 2,2n > {hcr_in_region}"
                )
            else:
                self.sp.print_and_run(f"cp {hcr} {hcr_in_region}")

            ground_truths_to_check[sample_id] = ground_truth_to_check_vcf
        return ground_truths_to_check

    def check(self):
        errors = []

        for sample_id in self.crams:
            print(f"Check consistency for {sample_id}:")
            crams = self.crams[sample_id]

            print("    crams = \n\t" + "\n\t".join(self.crams[sample_id]))
            print(f"    hcrs = {self.hcrs}")
            print(f"    ground_truth_vcfs = {self.ground_truth_vcfs}")

            for cram in crams:
                # Validate that each cram correlates to the ground-truth
                print("")
                hit_fractions = []
                max_hit_fraction = 0
                best_match = None
                match_to_expected_truth = None
                cram_base_name = os.path.basename(cram)

                called_vcf = f"{self.out_dir}/{cram_base_name}.calls.vcf.gz"
                local_bam = f"{self.out_dir}/{cram_base_name}.bam"
                self.sp.print_and_run(f"samtools view {cram} {self.region} -b -o {local_bam}")

                self.vc.call_variants(local_bam, called_vcf, self.region, min_af=self.min_af_snps)

                potential_error = f"{cram} - {sample_id} "
                for ground_truth_id, ground_truth_to_check_vcf in self.ground_truths_to_check.items():
                    hit_fraction, _, _ = self.vc.calc_hit_fraction(called_vcf, ground_truth_to_check_vcf)
                    if hit_fraction > max_hit_fraction:
                        max_hit_fraction = hit_fraction
                        best_match = ground_truth_id
                    hit_fractions.append(hit_fraction)
                    if sample_id == ground_truth_id and hit_fraction < self.min_hit_fraction_target:
                        match_to_expected_truth = hit_fraction
                        potential_error += f"does not match it's ground truth: hit_fraction={hit_fraction} "
                    elif sample_id != ground_truth_id and hit_fraction > self.min_hit_fraction_target:
                        potential_error += f"matched ground truth of {ground_truth_id}: hit_fraction={hit_fraction} "
                    print(f"{cram} - {sample_id} vs. {ground_truth_id} hit_fraction={hit_fraction}")
                if best_match != sample_id:
                    if match_to_expected_truth is None:
                        print(f"best_match={best_match} hit_fraction={max_hit_fraction}")
                    else:
                        potential_error += f"max_hit_fraction = {max(hit_fractions)}"
                if potential_error != f"{cram} - {sample_id} ":
                    errors.append(potential_error)
        if len(errors) > 0:
            raise RuntimeError("\n".join(errors))
