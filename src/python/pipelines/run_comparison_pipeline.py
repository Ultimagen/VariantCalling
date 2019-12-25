from python.pipelines import comparison_pipeline
from python.pipelines import vcf_pipeline_utils
import argparse



ap = argparse.ArgumentParser("Compare VCF to ground truth")
ap.add_argument("--n_parts", help='Number of parts that the VCF is split into', required=True, type=int)
ap.add_argument("--input_prefix", help="Prefix of the input file", required=True, type=str)
ap.add_argument("--output_file", help='Output h5 file', required=True, type=str)
ap.add_argument("--header_file", help="Desired VCF header of the file", required=True, type=str)
ap.add_argument("--gtr_vcf", help='Ground truth VCF file', required=True,type=str)
ap.add_argument("--cmp_intervals", help='Ranges on which to perform comparison', required=True, type=str)
ap.add_argument("--highconf_intervals", help='High confidence intervals', required=True, type=str )
ap.add_argument("--runs_intervals", help='Runs intervals', required=False, type=str, default=None)
ap.add_argument("--reference", help='Reference genome', required=True, type=str)
ap.add_argument("--aligned_bam", help='Aligned bam', required=True, type=str)
ap.add_argument("--call_sample_name", help='Name of the call sample', required=True, default='sm1')
ap.add_argument("--truth_sample_name", help='Name of the truth sample', required=True)
ap.add_argument("--find_thresholds", help='Should precision recall thresholds be found', default=False, action='store_true')
ap.add_argument("--filter_runs", help='Should variants on hmer runs be filtered out', default=False, action='store_true')
ap.add_argument("--output_suffix", help='Add suffix to the output file', required=False, default='', type=str)

args = ap.parse_args()
if args.filter_runs :
    results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix, 
        args.header_file, args.gtr_vcf, args.cmp_intervals, args.highconf_intervals, 
        args.runs_intervals, args.reference, args.call_sample_name, 
        args.truth_sample_name, args.find_thresholds, args.output_suffix)
else: 
    results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix, 
        args.header_file, args.gtr_vcf, args.cmp_intervals, args.highconf_intervals, 
        None, args.reference, args.call_sample_name, 
        args.truth_sample_name, args.find_thresholds, args.output_suffix)

if args.find_thresholds : 
    concordance = results[0]
else:
    concordance = results 

annotated_concordance = vcf_pipeline_utils.annotate_concordance(concordance, args.reference, args.aligned_bam, args.runs_intervals)

annotated_concordance.to_hdf(args.output_file, key="concordance")
if args.find_thresholds:
    results[1].to_hdf( args.output_file, key="results_calling")
    results[2].to_hdf( args.output_file, key="results_genotyping")




