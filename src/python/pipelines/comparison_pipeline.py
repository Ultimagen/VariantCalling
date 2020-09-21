from python.pipelines import vcf_pipeline_utils
from python import variant_filtering_utils
import pandas as pd
import shutil
from typing import Optional, List

TRUTH_FILE = "/home/ubuntu/proj/VariantCalling/data/giab/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.update_sd.vcf.gz"
CMP_INTERVALS = "/home/ubuntu/proj/VariantCalling/work/190614/interval.list"
HIGHCONF_INTERVALS = "/home/ubuntu/proj/VariantCalling/work/190614/GIAB_highconf.bed"
RUNS_INTERVALS = "/home/ubuntu/proj/VariantCalling/work/190724/runs.bed"
REFERENCE = "/home/ubuntu/proj/VariantCalling/data/genomes/hg19.fa"
CALL_SAMPLE = "NA12878"
TRUTH_SAMPLE = "HG001"


def pipeline(n_parts: int, input_prefix: str, header: str,
             truth_file: str= TRUTH_FILE, 
             cmp_intervals: str = CMP_INTERVALS,
             highconf_intervals: str = HIGHCONF_INTERVALS,
             runs_intervals: Optional[str] = RUNS_INTERVALS,
             ref_genome: str = REFERENCE,
             call_sample: str = CALL_SAMPLE,
             truth_sample: str = TRUTH_SAMPLE,
             find_thresholds: bool = True,
             output_suffix: str = '',
             ignore_filter: bool = False) -> tuple:
    '''Run comparison between the two sets of dataframes
    '''
    if not output_suffix:
        output_fn = input_prefix + ".vcf.gz"
    else:
        output_fn = input_prefix + f".{output_suffix}.vcf.gz"
    if n_parts > 0:
        vcf_pipeline_utils.combine_vcf(n_parts, input_prefix, output_fn)
    else:
        output_fn = input_prefix + ".vcf.gz"

    if not output_suffix:
        reheader_fn = input_prefix + ".rhdr.vcf.gz"
    else:
        reheader_fn = input_prefix + f".{output_suffix}.rhdr.vcf.gz"
    if header is not None:
        vcf_pipeline_utils.reheader_vcf(output_fn, header, reheader_fn)
    else:
        shutil.copy(output_fn, reheader_fn)
        shutil.copy(".".join((output_fn, "tbi")), ".".join((reheader_fn, "tbi")))

    if not output_suffix:
        select_intervals_fn = input_prefix + ".intsct.vcf.gz"
    else:
        select_intervals_fn = input_prefix + f".{output_suffix}.intsct.vcf.gz"
    vcf_pipeline_utils.intersect_with_intervals(reheader_fn, cmp_intervals, select_intervals_fn)

    output_prefix = select_intervals_fn[:select_intervals_fn.index(".intsct.vcf.gz")]


    vcfeval_concordance = True
    if vcfeval_concordance:
        vcf_pipeline_utils.run_vcfeval_concordance(select_intervals_fn, truth_file, output_prefix,
                                                    cmp_intervals, ref_genome, call_sample, truth_sample, ignore_filter)
        output_prefix = f'{output_prefix}.vcfeval_concordance'
    else:
        vcf_pipeline_utils.run_genotype_concordance(select_intervals_fn, truth_file, output_prefix,
                                                 cmp_intervals, call_sample, truth_sample, ignore_filter)
        output_prefix = f'{output_prefix}.genotype_concordance'

    vcf_pipeline_utils.filter_bad_areas(select_intervals_fn, highconf_intervals, runs_intervals)
    vcf_pipeline_utils.filter_bad_areas(output_prefix + ".vcf.gz",
                                        highconf_intervals, runs_intervals)
    if runs_intervals is not None:
        concordance = vcf_pipeline_utils.vcf2concordance(select_intervals_fn.replace("vcf.gz", "runs.vcf.gz"),
                                                         output_prefix + ".runs.vcf.gz")
    else:
        concordance = vcf_pipeline_utils.vcf2concordance(select_intervals_fn.replace("vcf.gz", "highconf.vcf.gz"),
                                                         output_prefix + ".highconf.vcf.gz")

    if find_thresholds:
        filtering_results = variant_filtering_utils.find_thresholds(concordance)
        filtering_results.index = pd.MultiIndex.from_tuples(filtering_results.index, names=['qual', 'sor'])

        filtering_results_gt = variant_filtering_utils.find_thresholds(concordance)
        filtering_results_gt.index = pd.MultiIndex.from_tuples(filtering_results_gt.index, names=['qual', 'sor'])

        return concordance, filtering_results, filtering_results_gt
    else:
        return concordance