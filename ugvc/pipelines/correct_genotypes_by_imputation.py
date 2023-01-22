#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Correct the genotypes of variants using imputation
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from tempfile import gettempdir

import numpy as np
from joblib import Parallel, delayed
from pysam import VariantFile
from simppl.simple_pipeline import SimplePipeline
from tqdm import tqdm

from ugvc import logger
from ugvc.utils.math_utils import phred, unphred
from ugvc.vcfbed.genotype import different_gt
from ugvc.vcfbed.vcftools import genotype_ordering


def get_parser() -> argparse.ArgumentParser:
    ap_var = argparse.ArgumentParser(
        prog="correct_genotypes_by_imputation.py", description="Correct the genotypes in a vcf using imputation"
    )
    ap_var.add_argument(
        "--input_vcf",
        help="VCF to be corrected",
        required=True,
        type=str,
    )
    ap_var.add_argument(
        "--chrom_to_cohort_vcfs_json",
        help="""json file that maps the chromosome names to vcf files with the reference cohort
(used only if the single_* arguments are not defined)""",
        required=False,
        type=str,
    )
    ap_var.add_argument(
        "--chrom_to_plink_json",
        help="""json file that maps the chromosome names to plink files with genomic maps
(used only if the single_* arguments are not defined)""",
        required=False,
        type=str,
    )
    ap_var.add_argument(
        "--single_chrom",
        help="""The chromosome to work on (suits a cromwell run. When used, chrom_to_cohort_vcfs_json
and chrom_to_plink_json should not be defined)""",
        required=False,
        type=str,
    )
    ap_var.add_argument(
        "--single_cohort_vcf",
        help="""The vcf with the reference cohort. Must be include the chromosome in single_chrom
(suits a cromwell run. When used chrom_to_cohort_vcfs_json and chrom_to_plink_json should not be defined)""",
        required=False,
        type=str,
    )
    ap_var.add_argument(
        "--single_genomic_map_plink",
        help="""The genomic map in plink format of the chromosome in single_chrom (suits a cromwell run.
When used chrom_to_cohort_vcfs_json and chrom_to_plink_json should not be defined)""",
        required=False,
        type=str,
    )
    ap_var.add_argument(
        "--epsilon",
        help="The weight given to the imputation when correcting the PL's. A number in the range (0,1)",
        required=True,
        type=float,
    )
    ap_var.add_argument(
        "--output_vcf",
        help="Name of output vcf file",
        required=True,
        type=str,
    )
    ap_var.add_argument(
        "--stats_file",
        help="Name of csv file to output counts of variants processed",
        required=False,
        type=str,
    )
    ap_var.add_argument(
        "--temp_dir",
        help="Path to temporary dir",
        required=False,
        type=str,
        default=gettempdir(),
    )
    ap_var.add_argument(
        "--threads_for_contig", help="Number of threads to use when looping across contigs", type=int, default=4
    )
    ap_var.add_argument(
        "--threads_beagle", help="Number of threads when running Beagle per contig", type=int, default=1
    )
    ap_var.add_argument(
        "--add_imp_effect",
        help="Add to the vcf the category of the effect of imputation on the variant (for debugging purposes)",
        action="store_true",
    )
    ap_var.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    return ap_var


def _command_to_subset_vcf(chrom: str, vcf: str, outfile: str) -> str:
    cmd = f"""set -eo pipefail
    bcftools view -O z -o {outfile} {vcf} {chrom} && \
    bcftools index -t {outfile}
    """
    return cmd


def _command_to_filter_gq_vcf(vcf: str, outfile: str) -> str:
    cmd = f"""set -eo pipefail
    bcftools view -f PASS {vcf} | \
    bcftools filter -i 'QUAL>20 && FORMAT/GQ[0]>20' | \
    bcftools view -O z -o {outfile} && \
    bcftools index -t {outfile}
    """
    return cmd


def _command_to_run_beagle(vcf: str, ref: str, gmap: str, outfile: str, nthreads: int = 1) -> str:
    outprefix = outfile.replace(".vcf.gz", "")
    cmd = f"""beagle -Xmx7g \
    gt={vcf} \
    ref={ref} \
    map={gmap} \
    out={outprefix} \
    nthreads={nthreads} window=100 && \
    bcftools index -t {outprefix}.vcf.gz
    """
    return cmd


def _command_to_filter_and_collpase_beagle(vcf: str, outfile: str) -> str:
    cmd = f"""set -eo pipefail
    bcftools view -i 'GT="alt"' {vcf} | \
    grep -v END | \
    bcftools norm --multiallelics + -o {outfile} -O z && \
    bcftools index -t {outfile}
    """
    return cmd


def _command_to_annotate_with_beagle(vcf: str, beagle_collapsed_vcf: str, outfile: str) -> str:
    cmd = f"""bcftools annotate --annotations {beagle_collapsed_vcf} \
    --columns INFO/IMP,INFO/DR2,FORMAT/DS -o {outfile} -O z  {vcf} && \
    bcftools index -t {outfile}
    """
    return cmd


def _f_along_allele(allele_i: int, gt_ar: np.ndarray, f_hom: np.ndarray, f_het: np.ndarray) -> np.ndarray:
    ar = (np.any(gt_ar == allele_i, axis=1) & (gt_ar[:, 0] == gt_ar[:, 1])) * f_hom[allele_i - 1] + (
        np.any(gt_ar == allele_i, axis=1) & (gt_ar[:, 0] != gt_ar[:, 1])
    ) * f_het[allele_i - 1]
    return ar


def _convert_ds_to_genotype_imputation_priors(
    ds_ar: np.ndarray, gt_ar: np.ndarray, num_alt: int, epsilon: float
) -> np.ndarray:
    # f_het and f_hom are the probabilities for het or hom, for each allele, capped by the epsilon
    # refer to https://ultimagen.atlassian.net/l/cp/K4RfDeJg for more details
    f_het = np.maximum(epsilon, np.minimum(2 - ds_ar, 1 - epsilon))
    f_hom = np.minimum(np.maximum(ds_ar, 1) - 1, 1 - epsilon)
    f_hom = np.maximum(epsilon, f_hom)
    # Arrange the probabilities f_het and f_hom into an array where the columns are the alleles, and
    # the rows are genotypes (the same as gt_ar)
    f_allele_ar = np.stack([_f_along_allele(i + 1, gt_ar, f_hom, f_het) for i in range(num_alt)], axis=1)
    # The probablity for each genotype is the max across all alleles of that genotype. When there are
    # missing values (i.e. where there is no imputation data on an allele), use the lowest cap.
    f_gt_ar = np.amax(np.nan_to_num(f_allele_ar, nan=epsilon), axis=1)
    # I don't change the PL for the hom-ref genotype, so the f is set to 1
    f_gt_ar[0] = 1

    return f_gt_ar


def modify_stats_with_imp(
    pl_tup: tuple[int], num_alt: int, ds_ar: np.ndarray, current_gt: tuple[int], epsilon: float
) -> tuple[tuple[int], int, tuple[int]]:
    """
    Modifies the PL values bases on the results from the imputation (the ALT dose, DS)
    Details of the calculation are explained in https://ultimagen.atlassian.net/l/cp/K4RfDeJg
    Input:
    pl_tup - A tuple with PLs (as in the vcf)
    num_alt - number of ALT alleles
    ds_ar - DS (allele dosage) values for each allele, stored in a numpy array.
            DS is an output of beagle and is defined as DS=P(RA) + 2P(AA).
            For bialleleic variants ds_ is typically 1.0 or 2.0 for het or hom variants, respectively.
            For tri-allelic variants it can be [1,1], [0,2] or [2,0], etc.
    current_gt - The genotype (GT) as in the vcf
    epsilon - The weight of the imputation in determining the new PL. A number between 0 and 1.
    Output:
    The new PLs, The new GQ and the new genotype (GT)
    """
    gt_ar = genotype_ordering(num_alt)
    f_gt_ar = _convert_ds_to_genotype_imputation_priors(ds_ar, gt_ar, num_alt, epsilon)
    # Modify the PLs with the imputation data
    pl_unphred = unphred(pl_tup)
    pl_f = pl_unphred * f_gt_ar
    # We don't want to change the balance between the ref and alt likelihoods, therefore
    # we normalize by the sum of all alt likelihoods
    pl_u_sum = np.sum(pl_unphred[1:])
    pl_f_sum = np.sum(pl_f[1:])
    pl_f_n = np.insert(pl_u_sum / pl_f_sum * pl_f[1:], 0, pl_unphred[0])
    phred_pl_f_n = phred(pl_f_n)
    # New genotype is determined by the lowest PL
    sorted_ids = np.argsort(phred_pl_f_n)
    current_gt_id = (gt_ar[:, 0] == current_gt[0]) & (gt_ar[:, 1] == current_gt[1])
    pl_new_gt = phred_pl_f_n[current_gt_id].tolist()[0]
    # If the lowest PL is the same as the PL assigned to the current genotype (a tie),then do
    # not change the GT (the ties are caused by round-offs in the original PL)
    if pl_new_gt == np.amin(phred_pl_f_n):
        new_gt = current_gt
    else:
        new_gt = tuple(gt_ar[sorted_ids, :][0].tolist())
    new_pl_ar = np.rint(phred_pl_f_n - min(phred_pl_f_n)).astype(int)
    new_gq = new_pl_ar[sorted_ids][1] - new_pl_ar[sorted_ids][0]

    return tuple(new_pl_ar.tolist()), new_gq.tolist(), new_gt


def add_imputation_to_vcf(
    beagle_anno_vcf: str,
    output_vcf: str,
    epsilon: float,
    region: tuple[str, int, int] | None = None,
    add_imp_effect: bool = False,
) -> dict:
    """
    Takes a vcf that is annotated with the Beagle results and outputs a new vcf with changes to the genotype, PLs, etc.
    Input:
    beagle_anno_vcf - Input vcf with Beagle annotations
    output_vcf - Name of output vcf
    epsilon - The weight of the imputation in determining the new PL. A number between 0 and 1.
    region - Process the vcf only in the specified region. A tuple (chrom,start,end). Experimental, for debugging.
    add_imp_effect - Add to the vcf the category of the effect of imputation on the variant.
                     These are the same categories used in the output counting stats.
                     For debugging purposes.
    Output:
    A dictionary with counts of variants that had imputation data, changed the genotype, etc.
    """

    counters = defaultdict(lambda: {"pass": 0, "has_non_ref_imp": 0, "imp_has_different_gt": 0, "changed_gt": 0})

    logger.info("Modifying PLs and writing to %s", output_vcf)
    with VariantFile(beagle_anno_vcf) as in_vcf_obj:
        # Modify header in order to save previous GT and PL
        gt_format = in_vcf_obj.header.formats["GT"]
        in_vcf_obj.header.formats.add(
            id="GT0",
            number=gt_format.number,
            type=gt_format.type,
            description=gt_format.description + " (DeepVariant output)",
        )

        gq_format = in_vcf_obj.header.formats["GQ"]
        in_vcf_obj.header.formats.add(
            id="GQ0",
            number=gq_format.number,
            type=gq_format.type,
            description=gq_format.description + " (DeepVariant output)",
        )

        pl_format = in_vcf_obj.header.formats["PL"]
        in_vcf_obj.header.formats.add(
            id="PL0",
            number=pl_format.number,
            type=pl_format.type,
            description=pl_format.description + " (DeepVariant output)",
        )
        if add_imp_effect:
            in_vcf_obj.header.info.add(
                id="IMP_EFFECT",
                type="String",
                number=1,
                description="Categories for the effect of imputation on the variant",
            )

        with VariantFile(output_vcf, "w", header=in_vcf_obj.header) as out_vcf_obj:
            if region is None:
                chrom = start = end = None
            else:
                chrom, start, end = region

            for rec in tqdm(in_vcf_obj.fetch(chrom, start, end)):
                if rec.filter.keys()[0] == "PASS" and sum(rec.samples[0]["GT"]) != 0:
                    num_alt = len(rec.alleles) - 1
                    # Update counters
                    vtype = rec.info["VARIANT_TYPE"] if num_alt == 1 else "multi"
                    counters[vtype]["pass"] += 1
                    if "DS" in rec.samples[0].keys():
                        current_gt = rec.samples[0]["GT"]
                        ds_ar = np.array(rec.samples[0]["DS"], dtype=np.float)
                        # Update counters
                        counters[vtype]["has_non_ref_imp"] += 1
                        cat_str = "has_non_ref_imp"
                        imp_is_hom = round(np.nanmax(ds_ar)) == 2
                        gt_is_hom = current_gt[0] == current_gt[1]
                        if imp_is_hom != gt_is_hom:
                            counters[vtype]["imp_has_different_gt"] += 1
                            cat_str += ",imp_has_different_gt"
                            # Note that in multi-allelics this check may be wrong, because
                            # it could be heterozygous in two different allels.
                        # Save the old GT, GQ and PL
                        rec.samples[0]["GT0"] = "|".join([str(a) for a in rec.samples[0]["GT"]])
                        rec.samples[0]["GQ0"] = rec.samples[0]["GQ"]
                        rec.samples[0]["PL0"] = rec.samples[0]["PL"]
                        # Modify the pl, gq and gt
                        pl_tup = rec.samples[0]["PL"]
                        new_pl, new_gq, new_gt = modify_stats_with_imp(pl_tup, num_alt, ds_ar, current_gt, epsilon)
                        rec.samples[0]["PL"] = new_pl
                        rec.samples[0]["GQ"] = new_gq
                        rec.samples[0]["GT"] = new_gt
                        if different_gt(new_gt, rec.samples[0]["GT0"]):
                            counters[vtype]["changed_gt"] += 1
                            cat_str += ",changed_gt"
                        if add_imp_effect:
                            print(cat_str)
                            rec.info["IMP_EFFECT"] = cat_str
                            print(rec)

                out_vcf_obj.write(rec)

        subprocess.check_output(f"bcftools index -t {output_vcf}", shell=True)

        return dict(counters)


def run(argv: list[str]):
    """Correct a vcf based on imputation"""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    jsons_are_defined = (args.chrom_to_cohort_vcfs_json is not None) and (args.chrom_to_plink_json is not None)
    singles_are_defined = (
        (args.single_chrom is not None)
        and (args.single_cohort_vcf is not None)
        and (args.single_genomic_map_plink is not None)
    )
    if jsons_are_defined and singles_are_defined:
        AssertionError(
            "You can define arguments for a single chromosome, or jsons for multiple chromosome, but not both"
        )
    elif jsons_are_defined and not singles_are_defined:
        logger.info("Running on multiple chromosomes. Will output concatenation of all chromosomes.")
        # Load the mapping between the chromosome names and the reference cohort files
        with open(args.chrom_to_cohort_vcfs_json, "r", encoding="utf-8") as json_file:
            chrom_to_cohort = json.load(json_file)
        # Load the mapping between the chromosome names and the reference cohort files
        with open(args.chrom_to_plink_json, "r", encoding="utf-8") as json_file:
            chrom_to_plink = json.load(json_file)
    elif singles_are_defined and not jsons_are_defined:
        logger.info(
            "Running on single chromosome (suits a cromwell run). Will output results just for %s", args.single_chrom
        )
        chrom_to_cohort = {args.single_chrom: args.single_cohort_vcf}
        chrom_to_plink = {args.single_chrom: args.single_genomic_map_plink}
    else:
        AssertionError(
            """You need to define three arguments for a single chromosome (single_chrom, single_cohort_vcf and
chrom_to_plink_json), or two arguments for multiple chromosome (chrom_to_cohort_vcfs_json and chrom_to_plink_json)"""
        )

    if args.stats_file is None:
        args.stats_file = args.output_vcf.replace(".vcf.gz", "") + "_counts.csv"

    sp = SimplePipeline(args.fc, args.lc, debug=args.d)

    subset_files = {chrom: os.path.join(args.temp_dir, f"subset.{chrom}.vcf.gz") for chrom in chrom_to_cohort.keys()}
    subset_vcf_cmds = [
        _command_to_subset_vcf(chrom, args.input_vcf, subset_files[chrom]) for chrom in chrom_to_cohort.keys()
    ]
    sp.run_parallel(subset_vcf_cmds, args.threads_for_contig)
    high_gq_files = {chrom: os.path.join(args.temp_dir, f"high_gq.{chrom}.vcf.gz") for chrom in chrom_to_cohort.keys()}
    high_gq_cmds = [
        _command_to_filter_gq_vcf(subset_files[chrom], high_gq_files[chrom]) for chrom in chrom_to_cohort.keys()
    ]
    sp.run_parallel(high_gq_cmds, args.threads_for_contig)
    beagle_files = {chrom: os.path.join(args.temp_dir, f"beagle.{chrom}.vcf.gz") for chrom in chrom_to_cohort.keys()}
    beagle_cmds = [
        _command_to_run_beagle(
            high_gq_files[chrom],
            chrom_to_cohort[chrom],
            chrom_to_plink[chrom],
            beagle_files[chrom],
            args.threads_beagle,
        )
        for chrom in chrom_to_cohort.keys()
    ]
    sp.run_parallel(beagle_cmds, args.threads_for_contig)
    beagle_collapsed_files = {
        chrom: os.path.join(args.temp_dir, f"beagle_collapsed.{chrom}.vcf.gz") for chrom in chrom_to_cohort.keys()
    }
    beagle_collapse_cmds = [
        _command_to_filter_and_collpase_beagle(beagle_files[chrom], beagle_collapsed_files[chrom])
        for chrom in chrom_to_cohort.keys()
    ]
    sp.run_parallel(beagle_collapse_cmds, args.threads_for_contig)
    beagle_anno_files = {
        chrom: os.path.join(args.temp_dir, f"beagle_anno.{chrom}.vcf.gz") for chrom in chrom_to_cohort.keys()
    }
    beagle_anno_cmds = [
        _command_to_annotate_with_beagle(subset_files[chrom], beagle_collapsed_files[chrom], beagle_anno_files[chrom])
        for chrom in chrom_to_cohort.keys()
    ]
    sp.run_parallel(beagle_anno_cmds, args.threads_for_contig)

    def run_add_imputation_to_vcf(chrom):
        out = add_imputation_to_vcf(
            beagle_anno_vcf=beagle_anno_files[chrom],
            output_vcf=os.path.join(args.temp_dir, f"add_imp.{chrom}.vcf.gz"),
            epsilon=args.epsilon,
            add_imp_effect=args.add_imp_effect,
        )
        return out

    counts_per_chrom = Parallel(n_jobs=args.threads_for_contig, verbose=10)(
        delayed(run_add_imputation_to_vcf)(chrom) for chrom in chrom_to_cohort.keys()
    )
    add_imp_files = {chrom: os.path.join(args.temp_dir, f"add_imp.{chrom}.vcf.gz") for chrom in chrom_to_cohort.keys()}
    # Sum counts from all chromosomes:
    counts = {"snp": {}, "h-indel": {}, "non-h-indel": {}, "multi": {}}
    for counts_i in counts_per_chrom:
        for var_type in counts_i.keys():
            for category in counts_i[var_type].keys():
                counts[var_type][category] = counts[var_type].get(category, 0) + counts_i[var_type][category]

    # Print the counts to csv file
    logger.info("Writing statistics to %s", args.stats_file)
    lines = {}
    with open(args.stats_file, "w", encoding="utf-8") as stats_file:
        var_types = counts.keys()
        stats_file.write("," + ",".join(var_types) + "\n")
        for vt in var_types:
            categories = counts[vt].keys()
            for cat in categories:
                lines[cat] = lines.get(cat, "") + "," + str(counts[vt][cat])
        for cat in categories:
            stats_file.write(cat + lines[cat] + "\n")

    if singles_are_defined:
        # Only a single chromosome is outputted (suits a cromwell run)
        sp.print_and_run(f"mv {add_imp_files[args.single_chrom]} {args.output_vcf}")
    else:
        # Concat files
        logger.info("Concatenating files")
        # with open(args.temp_dir) as regions_file:
        untouched_contigs_bed = os.path.join(args.temp_dir, "untouched_contigs.bed")
        untouched_contigs_vcf = os.path.join(args.temp_dir, "untouched_contigs.vcf.gz")
        contigs_info = {}
        with VariantFile(args.input_vcf) as in_vcf_obj:
            for k in in_vcf_obj.header.contigs.keys():
                contig = in_vcf_obj.header.contigs[k]
                contigs_info[contig.name] = {"id": contig.id, "length": contig.length}
        with open(untouched_contigs_bed, "w", encoding="utf-8") as regions_file:
            sorted_contigs = sorted(contigs_info.keys(), key=lambda k: contigs_info[k]["id"])
            for contig in sorted_contigs:
                if contig not in chrom_to_cohort.keys():
                    regions_file.write(contig + "\t1\t" + str(contigs_info[contig]["length"]) + "\n")
        # Extract chromosome without imputation data, and reheader to fit the header of vcfs after imputation
        sp.print_and_run(f"bcftools view -O z -o {untouched_contigs_vcf} -R {untouched_contigs_bed} {args.input_vcf}")
        untouched_contigs_rehead_vcf = os.path.join(args.temp_dir, "untouched_contigs_rehead.vcf.gz")
        header = os.path.join(args.temp_dir, "header.txt")
        add_imp_list = list(add_imp_files.values())
        cmd = f"""bcftools view -h {add_imp_list[0]} > {header} && \
        bcftools reheader -h {header} -o {untouched_contigs_rehead_vcf} {untouched_contigs_vcf}
        """
        sp.print_and_run(cmd)
        # Construct the command to concat the vcfs. Maintain the order as in the original vcf
        sorted_add_imp_files = [
            add_imp_files[chrom] for chrom in sorted(chrom_to_cohort.keys(), key=lambda k: contigs_info[k]["id"])
        ]
        vcfs_to_concat = sorted_add_imp_files + [untouched_contigs_rehead_vcf]
        sp.print_and_run(f"bcftools concat --naive -O z -o {args.output_vcf} {' '.join(vcfs_to_concat)}")
        sp.print_and_run(f"bcftools index -t {args.output_vcf}")


if __name__ == "__main__":
    run(sys.argv)
