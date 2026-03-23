from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simppl.simple_pipeline import SimplePipeline
from ugbio_cloud_utils.cloud_sync import optional_cloud_sync
from ugbio_core.vcf_utils import VcfUtils

from ugvc.comparison.variant_hit_fraction_caller import VariantHitFractionCaller


# pylint: disable=too-many-instance-attributes
class QuickFingerprinter:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        sample_crams: dict[str, list[str]],
        ground_truth_vcfs: dict[str, str],
        hcrs: dict[str, str],
        ref: str,
        region: str,
        min_af_snps: int,
        min_af_germline_snps: int,
        min_hit_fraction_target: float,
        add_aws_auth_command: bool,
        out_dir: str,
        sp: SimplePipeline,
        regions_bed: str | None = None,
        output_prefix: str = "fingerprint",
    ):
        self.crams = sample_crams
        self.ground_truth_vcfs = ground_truth_vcfs
        self.hcrs = hcrs
        self.ref = ref
        self.region = region
        self.regions_bed = regions_bed
        self.output_prefix = output_prefix
        self.out_dir = out_dir
        self.min_af_snps = min_af_snps
        self.min_af_germline_snps = min_af_germline_snps
        self.min_hit_fraction_target = min_hit_fraction_target
        self.sp = sp
        self.add_aws_auth_command = add_aws_auth_command
        self.vc = VariantHitFractionCaller(self.ref, self.out_dir, self.sp, self.min_af_snps, region)
        self.vpu = VcfUtils(self.sp)
        os.makedirs(out_dir, exist_ok=True)

        self.ground_truths_to_check = self.prepare_ground_truth()

    def prepare_ground_truth(self):
        ground_truths_to_check = {}
        region_bed = f"{self.out_dir}/region.bed"
        regions_bed_in_region = None
        if self.region != "":
            self.sp.print_and_run(f"echo {self.region} | sed 's/:/\t/' | sed 's/-/\t/' > {region_bed}")
        if self.regions_bed is not None and self.region != "":
            regions_bed_in_region = f"{self.out_dir}/regions_bed_in_region.bed"
            self.sp.print_and_run(
                f"bedtools intersect -a {self.regions_bed} -b {region_bed} | "
                f"sort -k 1,1 -k 2,2n > {regions_bed_in_region}"
            )

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
            self.vpu.index_vcf(ground_truth_in_hcr)
            if self.regions_bed is not None:
                regions_to_use = regions_bed_in_region if regions_bed_in_region is not None else self.regions_bed
                self.sp.print_and_run(
                    f"bcftools view {ground_truth_in_hcr} -R {regions_to_use} -Oz -o {ground_truth_to_check_vcf}"
                )
            else:
                self.sp.print_and_run(
                    f"bcftools view {ground_truth_in_hcr} -r {self.region} -Oz -o {ground_truth_to_check_vcf}"
                )
            self.vpu.index_vcf(ground_truth_to_check_vcf)

            if self.regions_bed is not None:
                self.sp.print_and_run(
                    f"bedtools intersect -a {hcr} -b {self.regions_bed} | "
                    f"sort -k 1,1 -k 2,2n > {hcr_in_region}"
                )
            elif self.region != "":
                self.sp.print_and_run(
                    f"bedtools intersect -a {hcr} -b {region_bed} | "
                    f"sort -k 1,1 -k 2,2n > {hcr_in_region}"
                )
            else:
                self.sp.print_and_run(f"cp {hcr} {hcr_in_region}")

            ground_truths_to_check[sample_id] = ground_truth_to_check_vcf
        return ground_truths_to_check

    def print(self, msg: str):
        self.output_file.write(msg + "\n")

    def check(self):
        errors = []
        all_results = []  # list of (sample_id, cram_base_name, ground_truth_ids, hit_fractions)
        with open(f"{self.out_dir}/quick_fingerprinting_results.txt", "w", encoding="utf-8") as of:
            self.output_file = of
            for sample_id in self.crams:
                self.print(f"Check consistency for {sample_id}:")
                crams = self.crams[sample_id]
                self.print("    crams = \n\t" + "\n\t".join(self.crams[sample_id]))
                self.print(f"    hcrs = {self.hcrs}")
                self.print(f"    ground_truth_vcfs = {self.ground_truth_vcfs}")

                for cram in crams:
                    # Validate that each cram correlates to the ground-truth
                    self.print("")
                    hit_fractions = []
                    max_hit_fraction = 0
                    best_match = None
                    match_to_expected_truth = None
                    cram_base_name = os.path.basename(cram)

                    called_vcf = f"{self.out_dir}/{cram_base_name}.calls.vcf.gz"
                    local_bam = f"{self.out_dir}/{cram_base_name}.bam"
    
                    if self.add_aws_auth_command:
                        self.sp.print_and_run(
                            f"eval $(aws configure export-credentials --format env-no-export) \
                                samtools view {cram} -T {self.ref} {self.region} -b -o {local_bam}"
                        )
                    else:
                        self.sp.print_and_run(f"samtools view {cram} -T {self.ref} {self.region} -b -o {local_bam}")

                    self.sp.print_and_run(f"samtools index {local_bam}")

                    self.vc.call_variants(local_bam, called_vcf, self.region, min_af=self.min_af_snps, regions_bed=self.regions_bed)

                    potential_error = f"{cram} - {sample_id} "
                    ground_truth_ids = []
                    cram_hit_fractions = []
                    for ground_truth_id, ground_truth_to_check_vcf in self.ground_truths_to_check.items():
                        hit_fraction, _, _ = self.vc.calc_hit_fraction(called_vcf, ground_truth_to_check_vcf)
                        ground_truth_ids.append(ground_truth_id)
                        cram_hit_fractions.append(hit_fraction)
                        if hit_fraction > max_hit_fraction:
                            max_hit_fraction = hit_fraction
                            best_match = ground_truth_id
                        hit_fractions.append(hit_fraction)
                        if sample_id == ground_truth_id and hit_fraction < self.min_hit_fraction_target:
                            match_to_expected_truth = hit_fraction
                            potential_error += f"does not match it's ground truth: hit_fraction={hit_fraction} "
                        elif sample_id != ground_truth_id and hit_fraction > self.min_hit_fraction_target:
                            potential_error += (
                                f"matched ground truth of {ground_truth_id}: hit_fraction={hit_fraction} "
                            )
                        self.print(f"{cram} - {sample_id} vs. {ground_truth_id} hit_fraction={hit_fraction}")
                    if best_match != sample_id:
                        if match_to_expected_truth is None:
                            self.print(f"{cram} best_match={best_match} hit_fraction={max_hit_fraction}")
                        else:
                            potential_error += f"max_hit_fraction = {max(hit_fractions)}"
                    if potential_error != f"{cram} - {sample_id} ":
                        errors.append(potential_error)

                    all_results.append((sample_id, cram_base_name, ground_truth_ids, cram_hit_fractions))

        self._save_combined_plot(all_results)
        if len(errors) > 0:
            raise RuntimeError("\n".join(errors))

    def _save_combined_plot(self, all_results: list[tuple]):
        """Save all CRAM hit-fraction bar charts as subplots in a single PNG."""
        n = len(all_results)
        if n == 0:
            return
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        n_gt = len(all_results[0][2])  # number of ground-truth samples
        subplot_w = max(4, n_gt * 0.9)
        fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_w * ncols, 5 * nrows), squeeze=False)
        for idx, (sample_id, cram_base_name, ground_truth_ids, hit_fractions) in enumerate(all_results):
            ax = axes[idx // ncols][idx % ncols]
            colors = []
            for gt_id, hf in zip(ground_truth_ids, hit_fractions):
                if gt_id == sample_id:
                    colors.append("green" if hf >= self.min_hit_fraction_target else "red")
                else:
                    colors.append("lightgrey")
            bars = ax.bar(ground_truth_ids, hit_fractions, color=colors)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Hit fraction")
            ax.set_xlabel("Ground truth")
            ax.set_title(f"Sample: {sample_id}\n{cram_base_name}", fontsize=8)
            ax.axhline(self.min_hit_fraction_target, color="red", linestyle="--", linewidth=1)
            for bar, val in zip(bars, hit_fractions):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(ground_truth_ids)))
            ax.set_xticklabels(ground_truth_ids, rotation=30, ha="right", fontsize=8)
        # hide unused axes
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle("Fingerprinting hit fractions (green = match above target, red = match below target, grey = non-match)", fontsize=10, y=1.01)
        plt.tight_layout()
        out_path = f"{self.out_dir}/{self.output_prefix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
