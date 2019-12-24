from . import comparison_pipeline
from . import vcf_pipeline_utils
import argparse



ap = argparse.ArgumentParser("Compare VCF to ground truth")
ap.add("--n_parts", 'Number of parts that the VCF is split into', required=True, type=int)
ap.add("--input_prefix", "Prefix of the input file", required=True, type=str)
ap.add("--output_file", 'Output h5 file', requried=True, type=str)
ap.add("--header_file", "Desired VCF header of the file", required=True, type=str)
ap.add("--gtr_vcf", 'Ground truth VCF file', required=True,type=str)
ap.add("--cmp_intervals", 'Ranges on which to perform comparison', required=True, type=str)
ap.add("--highconf_intervals", 'High confidence intervals', required=True, type=str )
ap.add("--runs_intervals", 'Runs intervals', required=False, type=str, default=None)
ap.add("--reference", 'Reference genome', required=True, type=str)
ap.add("--aligned_bam", 'Aligned bam', required=True, type=str)
ap.add("--call_sample_name", 'Name of the call sample', required=True, default='sm1')
ap.add("--truth_sample_name", 'Name of the truth sample', required=True)
ap.add("--find_thresholds", 'Should precision recall thresholds be found', type=bool, default=False, action='store_true')

args = ap.parse_args()
results = comparison_pipeline.pipeline(args.n_parts, args.input_prefix, 
    args.header_file, args.gtr_vcf, args.cmp_intervals, args.highconf_intervals, 
    args.runs_intervals, args.reference, args.call_sample_name, 
    args.truth_sample_name, args.find_thresholds)

if args.find_thresholds : 
    concordance = results[0]
else:
    concordance = results 

annotated_concordance = vcf_pipeline_utils.annotate_concordance(concordance, args.reference, args.aligned_bam)

annotated_concordance.to_hdf(args.output_file, key="concordance")
if args.find_thresholds:
    results[1].to_hdf( args.output_file, key="results_calling")
    results[2].to_hdf( args.output_file, key="results_genotyping")




