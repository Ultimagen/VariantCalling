import pathmagic
import python.pipelines.comparison_pipeline as comparison_pipeline
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import python.vcftools as vcftools
import argparse
import pandas as pd
import pysam
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from tempfile import NamedTemporaryFile
from shutil import copyfile
MIN_CONTIG_LENGTH = 100000


def _contig_concordance_annotate_reinterpretation(results, contig, reference, aligned_bam, annotate_intervals,
                                                  runs_intervals, hpol_filter_length_dist, flow_order,
                                                  base_name_outputfile, concordance_tool, disable_reinterpretation,
                                                  ignore_low_quality_fps):
    logger.info(f"Reading {contig}")
    concordance = vcf_pipeline_utils.vcf2concordance(
        results[0], results[1], concordance_tool, contig)
    annotated_concordance = vcf_pipeline_utils.annotate_concordance(
        concordance, reference, aligned_bam, annotate_intervals,
        runs_intervals, hmer_run_length_dist=hpol_filter_length_dist, flow_order=flow_order)

    if not disable_reinterpretation:
        annotated_concordance = vcf_pipeline_utils.reinterpret_variants(
            annotated_concordance, reference, ignore_low_quality_fps)
    logger.debug(f"{contig}: {annotated_concordance.shape}")
    annotated_concordance.to_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="run_comparison_pipeline.py", description="Compare VCF to ground truth")
    ap.add_argument(
        "--n_parts", help='Number of parts that the VCF is split into', required=True, type=int)
    ap.add_argument("--input_prefix", help="Prefix of the input file",
                    required=True, type=str)
    ap.add_argument("--output_file", help='Output h5 file',
                    required=True, type=str)
    ap.add_argument("--output_interval", help='Output bed file of intersected intervals',
                    required=True, type=str)
    ap.add_argument("--gtr_vcf", help='Ground truth VCF file',
                    required=True, type=str)
    ap.add_argument("--cmp_intervals", help='Ranges on which to perform comparison (bed/interval_list)',
                    required=False, type=str, default=None)
    ap.add_argument("--highconf_intervals",
                    help='High confidence intervals (bed/interval_list)', required=True, type=str)
    ap.add_argument("--runs_intervals", help='Runs intervals (bed/interval_list)',
                    required=False, type=str, default=None)
    ap.add_argument("--annotate_intervals", help='interval files for annotation (multiple possible)', required=False,
                    type=str, default=None, action='append')
    ap.add_argument("--reference", help='Reference genome',
                    required=True, type=str)
    ap.add_argument("--aligned_bam", help='Aligned bam',
                    required=False, default=None, type=str, action='append')
    ap.add_argument("--call_sample_name",
                    help='Name of the call sample', required=True, default='sm1')
    ap.add_argument("--truth_sample_name",
                    help='Name of the truth sample', required=True)
    ap.add_argument("--header_file", help="Desired header",
                    required=False, default=None)
    ap.add_argument("--filter_runs", help='Should variants on hmer runs be filtered out',
                    action='store_true')
    ap.add_argument("--hpol_filter_length_dist", nargs=2, type=int,
                    help='Length and distance to the hpol run to mark', default=[10, 10])
    ap.add_argument("--ignore_filter_status",
                    help="Ignore variant filter status", action='store_true')
    ap.add_argument("--flow_order", type=str,
                    help='Sequencing flow order (4 cycle)', required=False, default="TACG")
    ap.add_argument("--output_suffix", help='Add suffix to the output file',
                    required=False, default='', type=str)
    ap.add_argument("--concordance_tool", help='The concordance method to use (GC or VCFEVAL)',
                    required=False, default='VCFEVAL', type=str)
    ap.add_argument("--disable_reinterpretation",
                    help="Should re-interpretation be run", action="store_true")
    ap.add_argument("--is_mutect", help="Are the VCFs output of Mutect (false)",
                    action="store_true")
    ap.add_argument("--chr9_interval", help='Chr9 interval (bed/interval_list)', # hack for supporting the pipeline report
                    required=False, type=str, default=None)
    ap.add_argument("--n_jobs", help="n_jobs of parallel on contigs",type=int,
                    default=-1)
    ap.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.verbosity),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__ if __name__ != "__main__" else "run_comparison_pipeline")

    # intersect intervals and output as a bed file
    args_dict = {k: str(vars(args)[k]) for k in vars(args)}
    cmp_intervals = vcf_pipeline_utils.IntervalFile(args.cmp_intervals, args.reference)
    chr9_interval = vcf_pipeline_utils.IntervalFile(args.chr9_interval, args.reference)
    highconf_intervals = vcf_pipeline_utils.IntervalFile(args.highconf_intervals, args.reference)
    runs_intervals = vcf_pipeline_utils.IntervalFile(args.runs_intervals, args.reference)

    if cmp_intervals.is_none():# interval of highconf_intervals
        if chr9_interval.is_none():
            copyfile(highconf_intervals.as_bed_file(), args.output_interval)
        else: # length of ch9 and highconf_intervals intersected
            vcf_pipeline_utils.intersect_bed_files(chr9_interval.as_bed_file(), highconf_intervals.as_bed_file(), args.output_interval)
    else:
        if chr9_interval.is_none():
            vcf_pipeline_utils.intersect_bed_files(cmp_intervals.as_bed_file(), highconf_intervals.as_bed_file(),
                                                   args.output_interval)
        else: # intersect all the 3 intervals
            fp = NamedTemporaryFile()
            temp_file_path = fp.name
            vcf_pipeline_utils.intersect_bed_files(cmp_intervals.as_bed_file(), highconf_intervals.as_bed_file(), temp_file_path)
            vcf_pipeline_utils.intersect_bed_files(chr9_interval.as_bed_file(),temp_file_path, args.output_interval)

    pd.DataFrame(args_dict, index=[
        0]).to_hdf(args.output_file, key="input_args")


    if args.filter_runs:
        results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix,
                                               args.header_file, args.gtr_vcf, cmp_intervals.as_bed_file(),
                                               highconf_intervals.as_bed_file(),
                                               runs_intervals.as_bed_file(), args.reference, args.call_sample_name,
                                               args.truth_sample_name, args.output_suffix,
                                               args.ignore_filter_status,
                                               args.concordance_tool)
    else:
        results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix,
                                               args.header_file, args.gtr_vcf, cmp_intervals.as_bed_file(),
                                               highconf_intervals.as_bed_file(),
                                               None, args.reference, args.call_sample_name,
                                               args.truth_sample_name, args.output_suffix,
                                               args.ignore_filter_status,
                                               args.concordance_tool)

    # single interval-file concordance - will be saved in a single dataframe

    if not cmp_intervals.is_none():
        concordance = vcf_pipeline_utils.vcf2concordance(
            results[0], results[1], args.concordance_tool)
        concordance.to_hdf(
            "annotate_concordance_h5_input.hdf", key='concordance')
        annotated_concordance = vcf_pipeline_utils.annotate_concordance(
            concordance, args.reference, args.aligned_bam, args.annotate_intervals,
            runs_intervals.as_bed_file(), hmer_run_length_dist=args.hpol_filter_length_dist,
            flow_order=args.flow_order)

        if not args.disable_reinterpretation:
            annotated_concordance = vcf_pipeline_utils.reinterpret_variants(
                annotated_concordance, args.reference, ignore_low_quality_fps=args.is_mutect)
        annotated_concordance.to_hdf(args.output_file, key="concordance")
        vcftools.bed_files_output(annotated_concordance,
                                  args.output_file, mode='w', create_gt_diff=(not args.is_mutect))

    # whole-genome concordance - will be saved in dataframe per chromosome
    else:
        with pysam.VariantFile(results[0]) as vf:
            # we filter out short contigs to prevent huge files
            contigs = [x for x in vf.header.contigs if vf.header.contigs[
                x].length > MIN_CONTIG_LENGTH]

        base_name_outputfile = os.path.splitext(args.output_file)[0]
        Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
            delayed(_contig_concordance_annotate_reinterpretation)
            (results, contig, args.reference, args.aligned_bam, args.annotate_intervals, runs_intervals.as_bed_file(),
             args.hpol_filter_length_dist, args.flow_order, base_name_outputfile, args.concordance_tool,
             args.disable_reinterpretation, args.is_mutect)
            for contig in tqdm(contigs))

        # merge temp h5 files
        write_mode = 'w'

        # find columns and set the same header for empty dataframes
        for contig in contigs:
            h5_temp = pd.read_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)
            if h5_temp.shape == (0, 0):  # empty dataframes are dropped to save space
                continue
            else:
                df_columns = pd.DataFrame(columns=h5_temp.columns)
                break

        for contig in contigs:
            h5_temp = pd.read_hdf(f"{base_name_outputfile}{contig}.h5", key=contig)
            if h5_temp.shape == (0, 0):  # empty dataframes get default columns
                h5_temp = pd.concat((h5_temp, df_columns), axis=1)
            h5_temp.to_hdf(args.output_file, mode=write_mode, key=contig)
            write_mode = 'a'
            if contig == "chr9":
                h5_temp.to_hdf(args.output_file, mode=write_mode, key="concordance")
            os.remove(f"{base_name_outputfile}{contig}.h5")

        write_mode = 'w'
        for contig in contigs:
            annotated_concordance = pd.read_hdf(args.output_file, key=contig)
            vcftools.bed_files_output(
                annotated_concordance, args.output_file, mode=write_mode,
                create_gt_diff=(not args.is_mutect))
            write_mode = 'a'
