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
#    Compare UG callset to ground truth (preferentially using VCFEVAL as an engine)
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import logging
import os
import sys
from shutil import copyfile

import pandas as pd
import pysam
from joblib import Parallel, delayed
from simppl.simple_pipeline import SimplePipeline
from tqdm import tqdm

from ugvc import logger
from ugvc.comparison import vcf_pipeline_utils
from ugvc.comparison.comparison_pipeline import ComparisonPipeline
from ugvc.comparison.concordance_utils import read_hdf
from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.vcfbed import vcftools
from ugvc.vcfbed.interval_file import IntervalFile

MIN_CONTIG_LENGTH = 100000


def _contig_concordance_annotate_reinterpretation(
        # pylint: disable=too-many-arguments
        raw_calls_vcf,
        concordance_vcf,
        contig,
        reference,
        bw_high_quality,
        bw_all_quality,
        annotate_intervals,
        runs_intervals,
        hpol_filter_length_dist,
        flow_order,
        base_name_outputfile,
        disable_reinterpretation,
        ignore_low_quality_fps,
        scoring_field,
):
    logger.info("Reading %s", contig)
    concordance = vcf_pipeline_utils.vcf2concordance(
        raw_calls_vcf,
        concordance_vcf,
        contig,
        scoring_field=scoring_field,
    )

    annotated_concordance, _ = vcf_pipeline_utils.annotate_concordance(
        concordance,
        reference,
        bw_high_quality,
        bw_all_quality,
        annotate_intervals,
        runs_intervals,
        hmer_run_length_dist=hpol_filter_length_dist,
        flow_order=flow_order,
    )

    if not disable_reinterpretation:
        annotated_concordance = vcf_pipeline_utils.reinterpret_variants(
            annotated_concordance, reference, ignore_low_quality_fps
        )
    logger.debug("%s: %s", contig, annotated_concordance.shape)
    annotated_concordance.to_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)


def get_parser() -> argparse.ArgumentParser:
    ap_var = argparse.ArgumentParser(prog="run_comparison_pipeline.py", description="Compare VCF to ground truth")
    ap_var.add_argument(
        "--n_parts",
        help="Number of parts that the VCF is split into",
        required=True,
        type=int,
    )
    ap_var.add_argument("--input_prefix", help="Prefix of the input file", required=True, type=str)
    ap_var.add_argument("--output_file", help="Output h5 file", required=True, type=str)
    ap_var.add_argument(
        "--output_interval",
        help="Output bed file of intersected intervals",
        required=True,
        type=str,
    )
    ap_var.add_argument("--gtr_vcf", help="Ground truth VCF file", required=True, type=str)
    ap_var.add_argument(
        "--cmp_intervals",
        help="Ranges on which to perform comparison (bed/interval_list)",
        required=False,
        type=str,
        default=None,
    )
    ap_var.add_argument(
        "--highconf_intervals",
        help="High confidence intervals (bed/interval_list)",
        required=True,
        type=str,
    )
    ap_var.add_argument(
        "--runs_intervals",
        help="Runs intervals (bed/interval_list)",
        required=False,
        type=str,
        default=None,
    )
    ap_var.add_argument(
        "--annotate_intervals",
        help="interval files for annotation (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )
    ap_var.add_argument("--reference", help="Reference genome", required=True, type=str)
    ap_var.add_argument("--reference_dict", help="Reference genome dictionary", required=False, type=str)
    ap_var.add_argument(
        "--coverage_bw_high_quality",
        help="BigWig file with coverage only on high mapq reads",
        required=False,
        default=None,
        type=str,
        action="append",
    )
    ap_var.add_argument(
        "--coverage_bw_all_quality",
        help="BigWig file with coverage on all mapq reads",
        required=False,
        default=None,
        type=str,
        action="append",
    )
    ap_var.add_argument(
        "--call_sample_name",
        help="Name of the call sample",
        required=True,
        default="sm1",
    )
    ap_var.add_argument("--truth_sample_name", help="Name of the truth sample", required=True)
    ap_var.add_argument("--header_file", help="Desired header", required=False, default=None)
    ap_var.add_argument(
        "--filter_runs",
        help="Should variants on hmer runs be filtered out",
        action="store_true",
    )
    ap_var.add_argument(
        "--hpol_filter_length_dist",
        nargs=2,
        type=int,
        help="Length and distance to the hpol run to mark",
        default=[10, 10],
    )
    ap_var.add_argument("--ignore_filter_status", help="Ignore variant filter status", action="store_true")
    ap_var.add_argument(
        "--revert_hom_ref",
        help="For DeepVariant callsets - revert filtered hom_ref to het_ref for max recall calculation",
        action="store_true",
    )
    ap_var.add_argument(
        "--scoring_field",
        help="The pipeline expects a TREE_SCORE column in order to score the variants. If another field is \
        provided via scoring_field then its values will be copied to the TREE_SCORE column",
        required=False,
        default=None,
        type=str,
    )
    ap_var.add_argument(
        "--flow_order",
        type=str,
        help="Sequencing flow order (4 cycle)",
        required=False,
        default=DEFAULT_FLOW_ORDER,
    )
    ap_var.add_argument(
        "--output_suffix",
        help="Add suffix to the output file",
        required=False,
        default="",
        type=str,
    )
    ap_var.add_argument(
        "--disable_reinterpretation",
        help="Should re-interpretation be run",
        action="store_true",
    )
    ap_var.add_argument(
        "--special_chromosome",
        help="The chromosome that would be used for the \
        'concordance' dataframe (whole genome mode only)",
        default="chr9",
    )
    ap_var.add_argument("--is_mutect", help="Are the VCFs output of Mutect (false)", action="store_true")
    ap_var.add_argument("--n_jobs", help="n_jobs of parallel on contigs", type=int, default=-1)
    ap_var.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )
    return ap_var


def run(argv: list[str]):
    """Concordance between VCF and ground truth"""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))
    sp = SimplePipeline(args.fc, args.lc, debug=args.d, print_timing=True)
    vpu = vcf_pipeline_utils.VcfPipelineUtils(sp)

    cmp_intervals = IntervalFile(sp, args.cmp_intervals, args.reference, args.reference_dict)
    highconf_intervals = IntervalFile(sp, args.highconf_intervals, args.reference, args.reference_dict)
    runs_intervals = IntervalFile(sp, args.runs_intervals, args.reference, args.reference_dict)

    # intersect intervals and output as a bed file
    if cmp_intervals.is_none():  # interval of highconf_intervals
        logger.info(f"copy {args.highconf_intervals} to {args.output_interval}")
        copyfile(highconf_intervals.as_bed_file(), args.output_interval)
    else:
        vpu.intersect_bed_files(cmp_intervals.as_bed_file(), highconf_intervals.as_bed_file(), args.output_interval)

    args_dict = {k: str(vars(args)[k]) for k in vars(args)}
    pd.DataFrame(args_dict, index=[0]).to_hdf(args.output_file, key="input_args")

    runs_intervals_for_pipeline = runs_intervals if args.filter_runs else IntervalFile(sp)

    comparison_pipeline = ComparisonPipeline(
        vpu=vpu,
        n_parts=args.n_parts,
        input_prefix=args.input_prefix,
        truth_file=args.gtr_vcf,
        cmp_intervals=cmp_intervals,
        highconf_intervals=highconf_intervals,
        ref_genome=args.reference,
        call_sample=args.call_sample_name,
        truth_sample=args.truth_sample_name,
        output_file_name=args.output_file,
        header=args.header_file,
        runs_intervals=runs_intervals_for_pipeline,
        output_suffix=args.output_suffix,
        ignore_filter=args.ignore_filter_status,
        revert_hom_ref=args.revert_hom_ref
    )
    raw_calls_vcf, concordance_vcf = comparison_pipeline.run()

    # single interval-file concordance - will be saved in a single dataframe

    if not cmp_intervals.is_none():
        concordance_df = vcf_pipeline_utils.vcf2concordance(
            raw_calls_vcf,
            concordance_vcf,
            args.concordance_tool,
            scoring_field=args.scoring_field,
        )
        annotated_concordance_df, _ = vcf_pipeline_utils.annotate_concordance(
            concordance_df,
            args.reference,
            args.coverage_bw_high_quality,
            args.coverage_bw_all_quality,
            args.annotate_intervals,
            runs_intervals.as_bed_file(),
            hmer_run_length_dist=args.hpol_filter_length_dist,
            flow_order=args.flow_order,
        )

        if not args.disable_reinterpretation:
            annotated_concordance_df = vcf_pipeline_utils.reinterpret_variants(
                annotated_concordance_df,
                args.reference,
                ignore_low_quality_fps=args.is_mutect,
            )

        annotated_concordance_df.to_hdf(args.output_file, key="concordance", mode="a")
        # hack until we totally remove chr9
        annotated_concordance_df.to_hdf(args.output_file, key="comparison_result", mode="a")
        vcftools.bed_files_output(
            annotated_concordance_df,
            args.output_file,
            mode="w",
            create_gt_diff=(not args.is_mutect),
        )

    # whole-genome concordance - will be saved in dataframe per chromosome
    else:
        with pysam.VariantFile(raw_calls_vcf) as variant_file:
            # we filter out short contigs to prevent huge files
            contigs = [
                x for x in variant_file.header.contigs if variant_file.header.contigs[x].length > MIN_CONTIG_LENGTH
            ]

        base_name_outputfile = os.path.splitext(args.output_file)[0]
        Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
            delayed(_contig_concordance_annotate_reinterpretation)(
                raw_calls_vcf,
                concordance_vcf,
                contig,
                args.reference,
                args.coverage_bw_high_quality,
                args.coverage_bw_all_quality,
                args.annotate_intervals,
                runs_intervals.as_bed_file(),
                args.hpol_filter_length_dist,
                args.flow_order,
                base_name_outputfile,
                args.disable_reinterpretation,
                args.is_mutect,
                args.scoring_field,
            )
            for contig in tqdm(contigs)
        )

        # merge temp h5 files

        # find columns and set the same header for empty dataframes
        df_columns = None
        for contig in contigs:
            h5_temp = read_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)
            if h5_temp.shape == (0, 0):  # empty dataframes are dropped to save space
                continue
            df_columns = pd.DataFrame(columns=h5_temp.columns)
            break

        for contig in contigs:
            h5_temp = read_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)
            if h5_temp.shape == (0, 0):  # empty dataframes get default columns
                h5_temp = pd.concat((h5_temp, df_columns), axis=1)
            h5_temp.to_hdf(args.output_file, mode="a", key=contig)
            if contig == args.special_chromosome:
                h5_temp.to_hdf(args.output_file, mode="a", key="concordance")
            os.remove(f"{base_name_outputfile}{contig}.h5")

        write_mode = "w"
        for contig in contigs:
            annotated_concordance_df = read_hdf(args.output_file, key=contig)
            vcftools.bed_files_output(
                annotated_concordance_df,
                args.output_file,
                mode=write_mode,
                create_gt_diff=(not args.is_mutect),
            )
            write_mode = "a"


if __name__ == "__main__":
    run(sys.argv)
