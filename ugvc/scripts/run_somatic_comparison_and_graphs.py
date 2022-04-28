import pathmagic
import os
import argparse
import logging
import json
from os.path import join as pjoin
import subprocess
from datetime import datetime
from python.pipelines import run_comparison_pipeline
from python.pipelines import evaluate_concordance

ap = argparse.ArgumentParser(prog="run_somatic_comparison_and_graphs.py",
                             description="Create a performance analysis for mutect by running"
                                         "comaprison_pipeline and then create graphs out of it by evaluate_concordance")
## COMPARIOSN PARAMETERS
ap.add_argument("--n_parts", help='Number of parts that the VCF is split into',
                required=True, type=int)
ap.add_argument("--input_prefix", help="Prefix of the input file",
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
ap.add_argument("--reference_dict", help='Reference genome dictionary',
                required=False, type=str)
ap.add_argument("--call_sample_name",
                help='Name of the call sample', required=True, default='sm1')
ap.add_argument("--truth_sample_name",
                help='Name of the truth sample', required=True)
ap.add_argument("--hpol_filter_length_dist", nargs=2, type=int,
                help='Length and distance to the hpol run to mark', default=[10, 10])
ap.add_argument("--flow_order", type=str,
                help='Sequencing flow order (4 cycle)', required=False, default="TACG")
ap.add_argument("--disable_reinterpretation",
                help="Should re-interpretation be run", action="store_true")
ap.add_argument("--is_mutect", help="Are the VCFs output of Mutect (false)",
                action="store_true")
# ADDITIONAL PARAMETERS
ap.add_argument("--regions_bed", help="Regions to apply the simulation on (bed)",
                required=False, type=str)
ap.add_argument("--tumor_normal_analysis", help="Regions to apply the simulation on (bed)",
                required=False, type=bool, default=True)
ap.add_argument("--output_folder", help='Output folder',
                required=True, type=str)
ap.add_argument("--output_prefix", help='Output prefix',
                required=True, type=str)
ap.add_argument("--output_suffix", help='Add suffix to the output file',
                required=False, default='', type=str)
ap.add_argument("--gs_folder", help='Google cloud folder to upload the results to',
                required=False, type=str)
ap.add_argument("--json_db", help='Json file which contains all the runs that we ran, for future comparison',
                required=False, type=str)
ap.add_argument("--output_file", help='Output h5 file',
                required=True, type=str)
ap.add_argument("--output_interval", help='Output bed file of intersected intervals',
                required=True, type=str)

args = ap.parse_args()

### EXPLANATION FOR THE SCRIPT:
# After running mutect we want ot check performance.
# By using the gt_vcf and the cmp_interval bed file we compare the results from mutect to its gt file
# and then run evaluate_concordance in order to output the results.

# output files to be used:
# OUTPUT_gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}.vcf.gz -  the gt vcf for comparison
# OUTPUT_{prefix_cmp_interval}_no_problematic_positions_in_regions_only.bed - the cmp_interval file for comparison
###

logger = logging.getLogger(
    __name__ if __name__ != "__main__" else "create_somatic_gt_file")
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")



if args.regions_bed is not None:
    cmp_interval = pjoin(args.output_folder, f'cmp_interval_in_regions_only.bed')
    with open(cmp_interval,
              "w") as outfile:
        cmd = ['bedtools','intersect','-a',args.cmp_intervals, '-b',args.regions_bed]
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd, stdout=outfile)
else:
    cmp_interval = args.cmp_intervals

# remove multiallelic
## In case we have normal sample, we have 4 alleles.
# In tumor only sample we have only 2 alleles
n_alleles = 4 if args.tumor_normal_analysis else 2
cmd1 = ['bcftools','+fill-tags',f'{args.input_prefix}.vcf.gz','--','-t','AN']
cmd2 = ['bcftools', 'filter', '-Oz' ,'-o' ,pjoin(args.output_folder,'tmp_no_mutliallelic.vcf.gz') ,'-e', f"INFO/AN>{n_alleles}", '-']
logger.info(" ".join(cmd1))
ps = subprocess.run(cmd1, check=True, capture_output=True)
logger.info(" ".join(cmd2))
processNames = subprocess.run(cmd2,
                              input=ps.stdout, capture_output=True)

cmd = ['bcftools', 'index', '-t', pjoin(args.output_folder,'tmp_no_mutliallelic.vcf.gz')]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

comparison_output_h5 = pjoin(args.output_folder,args.output_prefix+args.output_suffix+'.h5')
comparison_output_interval = pjoin(args.output_folder,args.output_prefix+args.output_suffix+'.bed')
# run_comparison_pipeline
cmd = ['--n_parts',str(args.n_parts),
       '--input_prefix',pjoin(args.output_folder,'tmp_no_mutliallelic'),
       '--output_file',args.output_file,
       '--output_interval',args.output_interval,
       '--gtr_vcf',args.gtr_vcf,
       '--cmp_intervals',cmp_interval,
       '--highconf_intervals',args.highconf_intervals,
       '--runs_intervals',args.runs_intervals,
       '--reference',args.reference,
       '--reference_dict',args.reference_dict,
       '--call_sample_name',args.call_sample_name,
       '--flow_order',args.flow_order,
       '--hpol_filter_length_dist', '12','10',
       '--truth_sample_name',args.truth_sample_name]

for x in args.annotate_intervals:
    cmd.extend(['--annotate_intervals',x])

if args.disable_reinterpretation:
    cmd.append('--disable_reinterpretation')

logger.info(" ".join(cmd))
run_comparison_pipeline.main(cmd)

evaluate_prefix = pjoin(args.output_folder,'evaluate_'+args.output_prefix+args.output_suffix)
## EVALUATE CONCORDANCE
cmd = [
    '--input_file',args.output_file,
    '--output_prefix',evaluate_prefix
]
evaluate_concordance.run(cmd)


## UPLOAD THE RESULTS TO GS BUCKET
# upload evaluation h5 file
if args.gs_folder is not None:
    cmd = ['gsutil','cp',evaluate_prefix+'.stats.csv',args.gs_folder]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    # upload evaluation csv file
    cmd = ['gsutil','cp',evaluate_prefix+'.h5',args.gs_folder]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

## SAVE THE FILES IN A LOCAL DB
if args.json_db is not None:
    now = datetime.now()
    entry = {
        'input_prefix':args.input_prefix,
        'cmp_output_h5':comparison_output_h5,
        'cmp_output_interval':comparison_output_interval,
        'gtr_vcf':args.gtr_vcf,
        'cmp_intervals':args.cmp_intervals,
        'highconf_intervals':args.highconf_intervals,
        'runs_intervals':args.runs_intervals,
        'annotate_intervals':args.annotate_intervals,
        'reference':args.reference,
        'reference_dict':args.reference_dict,
        'call_sample_name':args.call_sample_name,
        'truth_sample_name':args.truth_sample_name,
        'flow_order':args.flow_order,
        'disable_reinterpretation':args.disable_reinterpretation,
        'is_mutect':args.is_mutect,
        'regions_bed':args.regions_bed,
        'tumor_normal_analysis':args.tumor_normal_analysis,
        'output_folder':args.output_folder,
        'gs_folder':args.gs_folder,
        'date':now.strftime("%d/%m/%Y %H:%M:%S")
    }

    if not os.path.isfile(args.json_db):
        a = []
        a.append(entry)
        with open(args.json_db, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(args.json_db) as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(entry)
        with open(args.json_db, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

