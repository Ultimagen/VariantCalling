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
MAX_CONTIG_LENGTH = 100000

logger = logging.getLogger(__name__)

ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


def _contig_concordance_annotate_reinterpretation(results, contig, reference, aligned_bam, annotate_intervals,
                                                 runs_intervals, hpol_filter_length_dist, base_name_outputfile,
                                                 concordance_tool, disable_reinterpretation, ignore_low_quality_fps):
    logger.info(f"Reading {contig}")
    concordance = vcf_pipeline_utils.vcf2concordance(
        results[0], results[1], concordance_tool, contig)
    annotated_concordance = vcf_pipeline_utils.annotate_concordance(
        concordance, reference, aligned_bam, annotate_intervals,
        runs_intervals, hmer_run_length_dist=hpol_filter_length_dist)

    if not disable_reinterpretation:
        annotated_concordance = vcf_pipeline_utils.reinterpret_variants(
            annotated_concordance, reference, ignore_low_quality_fps)

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
    ap.add_argument("--gtr_vcf", help='Ground truth VCF file',
                    required=True, type=str)
    ap.add_argument("--cmp_intervals", help='Ranges on which to perform comparison',
                    required=False, type=str, default=None)
    ap.add_argument("--highconf_intervals",
                    help='High confidence intervals', required=True, type=str)
    ap.add_argument("--runs_intervals", help='Runs intervals',
                    required=False, type=str, default=None)
    ap.add_argument("--annotate_intervals", help='interval files for annotation (multiple possible)', required=False,
                    type=str, default=None, action='append')
    ap.add_argument("--reference", help='Reference genome',
                    required=True, type=str)
    ap.add_argument("--aligned_bam", help='Aligned bam',
                    required=False, default=None, type=str)
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
    ap.add_argument("--output_suffix", help='Add suffix to the output file',
                    required=False, default='', type=str)
    ap.add_argument("--concordance_tool", help='The concordance method to use (GC or VCFEVAL)',
                    required=False, default='VCFEVAL', type=str)
    ap.add_argument("--disable_reinterpretation",
                    help="Should re-interpretation be run", action="store_true")
    ap.add_argument("--is_mutect", help="Are the VCFs output of Mutect (false)",
                    action="store_true")
    ap.add_argument("--n_jobs", help="n_jobs of parallel on contigs",
                    default=-1)

    args = ap.parse_args()

    pd.DataFrame({k: str(vars(args)[k]) for k in vars(args)}, index=[
        0]).to_hdf(args.output_file, key="input_args")


    if args.filter_runs:
        results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix,
                                               args.header_file, args.gtr_vcf, args.cmp_intervals, args.highconf_intervals,
                                               args.runs_intervals, args.reference, args.call_sample_name,
                                               args.truth_sample_name, args.output_suffix,
                                               args.ignore_filter_status,
                                               args.concordance_tool)
    else:
        results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix,
                                               args.header_file, args.gtr_vcf, args.cmp_intervals, args.highconf_intervals,
                                               None, args.reference, args.call_sample_name,
                                               args.truth_sample_name, args.output_suffix,
                                               args.ignore_filter_status,
                                               args.concordance_tool)

    # single interval-file concordance - will be saved in a single dataframe
    if args.cmp_intervals is not None:
        concordance = vcf_pipeline_utils.vcf2concordance(
            results[0], results[1], args.concordance_tool)
        concordance.to_hdf("annotate_concordance_h5_input.hdf",key='concordance')
        annotated_concordance = vcf_pipeline_utils.annotate_concordance(
            concordance, args.reference, args.aligned_bam, args.annotate_intervals,
            args.runs_intervals, hmer_run_length_dist=args.hpol_filter_length_dist)

        if not args.disable_reinterpretation:
            annotated_concordance = vcf_pipeline_utils.reinterpret_variants(
                annotated_concordance, args.reference, ignore_low_quality_fps = args.is_mutect)
        annotated_concordance.to_hdf(args.output_file, key="concordance")
        vcftools.bed_files_output(annotated_concordance,
                                  args.output_file, mode='w', create_gt_diff=(not args.is_mutect))

    # whole-genome concordance - will be saved in dataframe per chromosome
    else:
        with pysam.VariantFile(results[0]) as vf:
            # we filter out short contigs to prevent huge files
            contigs = [x for x in vf.header.contigs if vf.header.contigs[
                x].length > MAX_CONTIG_LENGTH]
        write_mode = 'w'

        base_name_outputfile = os.path.splitext(args.output_file)[0]
        Parallel(n_jobs=args.n_jobs,max_nbytes=None)(
            delayed(_contig_concordance_annotate_reinterpretation)
            (results, contig, args.reference, args.aligned_bam, args.annotate_intervals, args.runs_intervals,
                            args.hpol_filter_length_dist, base_name_outputfile, args.concordance_tool, 
                            args.disable_reinterpretation, args.is_mutect)
            for contig in tqdm(contigs))
        # merge temp h5 files
        hdfStore = pd.HDFStore(args.output_file, mode = 'w')
        for contig in contigs:
            h5_temp = pd.read_hdf(f"{base_name_outputfile}{contig}.h5", key = contig)
            hdfStore.put(contig, h5_temp)
            os.remove(f"{base_name_outputfile}{contig}.h5")

        for contig in contigs:
            annotated_concordance = hdfStore.get(contig)
            vcftools.bed_files_output(
                annotated_concordance, args.output_file, mode=write_mode,
                create_gt_diff=(not args.is_mutect))
        hdfStore.close()