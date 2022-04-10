#!/env/python
import argparse
import logging
import os
import sys
from shutil import copyfile
from typing import List

import pandas as pd
import pysam
from joblib import Parallel, delayed
from tqdm import tqdm

import ugvc.comparison.comparison_pipeline as comparison_pipeline
import ugvc.comparison.vcf_pipeline_utils as vcf_pipeline_utils
import ugvc.vcfbed.vcftools as vcftools
from ugvc.comparison.concordance_utils import read_hdf
from ugvc.dna.format import DEFAULT_FLOW_ORDER

MIN_CONTIG_LENGTH = 100000


def _contig_concordance_annotate_reinterpretation(
    results,
    contig,
    reference,
    bw_high_quality,
    bw_all_quality,
    annotate_intervals,
    runs_intervals,
    hpol_filter_length_dist,
    flow_order,
    base_name_outputfile,
    concordance_tool,
    disable_reinterpretation,
    ignore_low_quality_fps,
):
    logger = logging.getLogger(
        __name__ if __name__ != "__main__" else "run_comparison_pipeline"
    )
    logger.info(f"Reading {contig}")
    concordance = vcf_pipeline_utils.vcf2concordance(
        results[0], results[1], concordance_tool, contig
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
    logger.debug(f"{contig}: {annotated_concordance.shape}")
    annotated_concordance.to_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_comparison_pipeline.py", description="Compare VCF to ground truth"
    )
    ap.add_argument(
        "--n_parts",
        help="Number of parts that the VCF is split into",
        required=True,
        type=int,
    )
    ap.add_argument(
        "--input_prefix", help="Prefix of the input file", required=True, type=str
    )
    ap.add_argument("--output_file", help="Output h5 file", required=True, type=str)
    ap.add_argument(
        "--output_interval",
        help="Output bed file of intersected intervals",
        required=True,
        type=str,
    )
    ap.add_argument("--gtr_vcf", help="Ground truth VCF file", required=True, type=str)
    ap.add_argument(
        "--cmp_intervals",
        help="Ranges on which to perform comparison (bed/interval_list)",
        required=False,
        type=str,
        default=None,
    )
    ap.add_argument(
        "--highconf_intervals",
        help="High confidence intervals (bed/interval_list)",
        required=True,
        type=str,
    )
    ap.add_argument(
        "--runs_intervals",
        help="Runs intervals (bed/interval_list)",
        required=False,
        type=str,
        default=None,
    )
    ap.add_argument(
        "--annotate_intervals",
        help="interval files for annotation (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )
    ap.add_argument("--reference", help="Reference genome", required=True, type=str)
    ap.add_argument(
        "--reference_dict", help="Reference genome dictionary", required=False, type=str
    )
    ap.add_argument(
        "--coverage_bw_high_quality",
        help="BigWig file with coverage only on high mapq reads",
        required=False,
        default=None,
        type=str,
        action="append",
    )
    ap.add_argument(
        "--coverage_bw_all_quality",
        help="BigWig file with coverage on all mapq reads",
        required=False,
        default=None,
        type=str,
        action="append",
    )
    ap.add_argument(
        "--call_sample_name",
        help="Name of the call sample",
        required=True,
        default="sm1",
    )
    ap.add_argument(
        "--truth_sample_name", help="Name of the truth sample", required=True
    )
    ap.add_argument(
        "--header_file", help="Desired header", required=False, default=None
    )
    ap.add_argument(
        "--filter_runs",
        help="Should variants on hmer runs be filtered out",
        action="store_true",
    )
    ap.add_argument(
        "--hpol_filter_length_dist",
        nargs=2,
        type=int,
        help="Length and distance to the hpol run to mark",
        default=[10, 10],
    )
    ap.add_argument(
        "--ignore_filter_status",
        help="Ignore variant filter status",
        action="store_true",
    )
    ap.add_argument(
        "--flow_order",
        type=str,
        help="Sequencing flow order (4 cycle)",
        required=False,
        default=DEFAULT_FLOW_ORDER,
    )
    ap.add_argument(
        "--output_suffix",
        help="Add suffix to the output file",
        required=False,
        default="",
        type=str,
    )
    ap.add_argument(
        "--concordance_tool",
        help="The concordance method to use (GC or VCFEVAL)",
        required=False,
        default="VCFEVAL",
        type=str,
    )
    ap.add_argument(
        "--disable_reinterpretation",
        help="Should re-interpretation be run",
        action="store_true",
    )
    ap.add_argument(
        "--is_mutect", help="Are the VCFs output of Mutect (false)", action="store_true"
    )
    ap.add_argument(
        "--n_jobs", help="n_jobs of parallel on contigs", type=int, default=-1
    )
    ap.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )
    return ap.parse_args(argv)


def run(argv: List[str]):
    "Concordance between VCF and ground truth"
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.verbosity),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(
        __name__ if __name__ != "__main__" else "run_comparison_pipeline"
    )

    cmp_intervals = vcf_pipeline_utils.IntervalFile(
        args.cmp_intervals, args.reference, args.reference_dict
    )
    highconf_intervals = vcf_pipeline_utils.IntervalFile(
        args.highconf_intervals, args.reference, args.reference_dict
    )
    runs_intervals = vcf_pipeline_utils.IntervalFile(
        args.runs_intervals, args.reference, args.reference_dict
    )

    # intersect intervals and output as a bed file
    if cmp_intervals.is_none():  # interval of highconf_intervals
        copyfile(highconf_intervals.as_bed_file(), args.output_interval)
    else:
        vcf_pipeline_utils.intersect_bed_files(
            cmp_intervals.as_bed_file(),
            highconf_intervals.as_bed_file(),
            args.output_interval,
        )
    args_dict = {k: str(vars(args)[k]) for k in vars(args)}
    pd.DataFrame(args_dict, index=[0]).to_hdf(args.output_file, key="input_args")

    if args.filter_runs:
        results = comparison_pipeline.pipeline(
            args.n_parts,
            args.input_prefix,
            args.gtr_vcf,
            cmp_intervals,
            highconf_intervals,
            args.reference,
            args.call_sample_name,
            args.truth_sample_name,
            None,
            args.output_file,
            args.header_file,
            runs_intervals,
            args.output_suffix,
            args.ignore_filter_status,
            args.concordance_tool,
        )
    else:
        results = comparison_pipeline.pipeline(
            args.n_parts,
            args.input_prefix,
            args.gtr_vcf,
            cmp_intervals,
            highconf_intervals,
            args.reference,
            args.call_sample_name,
            args.truth_sample_name,
            None,
            args.output_file,
            args.header_file,
            vcf_pipeline_utils.IntervalFile(),
            args.output_suffix,
            args.ignore_filter_status,
            args.concordance_tool,
        )

    # single interval-file concordance - will be saved in a single dataframe

    if not cmp_intervals.is_none():
        concordance = vcf_pipeline_utils.vcf2concordance(
            results[0], results[1], args.concordance_tool
        )
        annotated_concordance, _ = vcf_pipeline_utils.annotate_concordance(
            concordance,
            args.reference,
            args.coverage_bw_high_quality,
            args.coverage_bw_all_quality,
            args.annotate_intervals,
            runs_intervals.as_bed_file(),
            hmer_run_length_dist=args.hpol_filter_length_dist,
            flow_order=args.flow_order,
        )

        if not args.disable_reinterpretation:
            annotated_concordance = vcf_pipeline_utils.reinterpret_variants(
                annotated_concordance,
                args.reference,
                ignore_low_quality_fps=args.is_mutect,
            )
        annotated_concordance.to_hdf(args.output_file, key="concordance", mode="a")
        # hack until we totally remove chr9
        annotated_concordance.to_hdf(
            args.output_file, key="comparison_result", mode="a"
        )
        vcftools.bed_files_output(
            annotated_concordance,
            args.output_file,
            mode="w",
            create_gt_diff=(not args.is_mutect),
        )

    # whole-genome concordance - will be saved in dataframe per chromosome
    else:
        with pysam.VariantFile(results[0]) as vf:
            # we filter out short contigs to prevent huge files
            contigs = [
                x
                for x in vf.header.contigs
                if vf.header.contigs[x].length > MIN_CONTIG_LENGTH
            ]

        base_name_outputfile = os.path.splitext(args.output_file)[0]
        Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
            delayed(_contig_concordance_annotate_reinterpretation)(
                results,
                contig,
                args.reference,
                args.coverage_bw_high_quality,
                args.coverage_bw_all_quality,
                args.annotate_intervals,
                runs_intervals.as_bed_file(),
                args.hpol_filter_length_dist,
                args.flow_order,
                base_name_outputfile,
                args.concordance_tool,
                args.disable_reinterpretation,
                args.is_mutect,
            )
            for contig in tqdm(contigs)
        )

        # merge temp h5 files

        # find columns and set the same header for empty dataframes
        for contig in contigs:
            h5_temp = read_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)
            if h5_temp.shape == (0, 0):  # empty dataframes are dropped to save space
                continue
            else:
                df_columns = pd.DataFrame(columns=h5_temp.columns)
                break

        for contig in contigs:
            h5_temp = read_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)
            if h5_temp.shape == (0, 0):  # empty dataframes get default columns
                h5_temp = pd.concat((h5_temp, df_columns), axis=1)
            h5_temp.to_hdf(args.output_file, mode="a", key=contig)
            if contig == "chr9":
                h5_temp.to_hdf(args.output_file, mode="a", key="concordance")
            os.remove(f"{base_name_outputfile}{contig}.h5")

        write_mode = "w"
        for contig in contigs:
            annotated_concordance = read_hdf(args.output_file, key=contig)
            vcftools.bed_files_output(
                annotated_concordance,
                args.output_file,
                mode=write_mode,
                create_gt_diff=(not args.is_mutect),
            )
            write_mode = "a"


if __name__ == "__main__":
    run(sys.argv[1:])
