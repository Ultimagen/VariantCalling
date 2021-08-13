from python.pipelines import vcf_pipeline_utils
import shutil
from os.path import join as pjoin
from os.path import basename
from typing import Optional, Tuple

TRUTH_FILE = "/home/ubuntu/proj/VariantCalling/data/giab/HG001_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.update_sd.vcf.gz"
CMP_INTERVALS = "/home/ubuntu/proj/VariantCalling/work/190614/interval.list"
HIGHCONF_INTERVALS = "/home/ubuntu/proj/VariantCalling/work/190614/GIAB_highconf.bed"
RUNS_INTERVALS = "/home/ubuntu/proj/VariantCalling/work/190724/runs.bed"
REFERENCE = "/home/ubuntu/proj/VariantCalling/data/genomes/hg19.fa"
CALL_SAMPLE = "NA12878"
TRUTH_SAMPLE = "HG001"
CONCORDANCE_TOOL = "VCFEVAL"


def pipeline(n_parts: int, input_prefix: str, 
            output_dir: str, 
            header: Optional[str] = None,
             truth_file: str = TRUTH_FILE,
             cmp_intervals: str = CMP_INTERVALS,
             highconf_intervals: str = HIGHCONF_INTERVALS,
             runs_intervals: Optional[str] = RUNS_INTERVALS,
             ref_genome: str = REFERENCE,
             call_sample: str = CALL_SAMPLE,
             truth_sample: str = TRUTH_SAMPLE,
             output_suffix: str = None,
             ignore_filter: bool = False,
             concordance_tool: str = CONCORDANCE_TOOL) -> Tuple[str, str]:
    '''Run comparison between the two sets of calls: input_prefix and truth_file. Creates
    a combined call file and a concordance VCF by either of the concordance tools

    Parameters
    ----------
    n_parts : int
        For input VCF split into number of parts - specifiy the number of parts. Specify
        zero for complete VCF
    input_prefix : str
        Input prefix for the vcf. If the vcf is split into multiple parts, the script
        will look for <input_prefix>.1.vcf, <input_prefix>.2.vcf etc. For the non-split VCF
        will look for <input_prefix>.vcf.gz
    output_dir : str
        Output directory for the output 
    header : str, optional
        for backward compatibility - to be able to change the header of the VCF. Default None
    truth_file : str, optional
        Truth calls file
    cmp_intervals : str, optional
        interval_list file over which to do comparison (e.g. chr9)
    highconf_intervals : str, optional
        high confidence intervals for the ground truth (BED)
    runs_intervals : str, optional
        Hompolymer runs annotation (BED)
    ref_genome : str, optional
        Reference genome FASTA
    call_sample : str, optional
        Name of the calls sample
    truth_sample : str, optional
        Name of the truth sample
    output_suffix : str, optional
        Suffix for the output file name (e.g. chr9) -
        otherwise the output file nams are starting with the input prefix
    ignore_filter : bool, optional
        Should the filter status **of calls only** be ignored. Filter status of truth is always
        taken into account
    concordance_tool : str, optional
        GC - GenotypeConcordance (picard) or VCFEVAL (default)

    Returns
    -------
    Tuple[str, str]
    '''
    input_prefix_basename = basename(input_prefix)

    if not output_suffix:
        output_fn = pjoin(output_dir, input_prefix_basename + ".vcf.gz")
    else:
        output_fn = pjoin(output_dir, input_prefix_basename + f".{output_suffix}.vcf.gz")
    if n_parts > 0:
        vcf_pipeline_utils.combine_vcf(n_parts, input_prefix, output_fn)
    else:
        output_fn = input_prefix + ".vcf.gz"

    if not output_suffix:
        reheader_fn = pjoin(output_dir, input_prefix_basename + ".rhdr.vcf.gz")
    else:
        reheader_fn = pjoin(output_dir, input_prefix_basename + f".{output_suffix}.rhdr.vcf.gz")

    if header is not None:
        vcf_pipeline_utils.reheader_vcf(output_fn, header, reheader_fn)
    else:
        shutil.copy(output_fn, reheader_fn)
        shutil.copy(".".join((output_fn, "tbi")), ".".join((reheader_fn, "tbi")))

    if not output_suffix:
        select_intervals_fn = pjoin(output_dir, input_prefix_basename + ".intsct.vcf.gz")
    else:
        select_intervals_fn = pjoin(output_dir, input_prefix_basename + f".{output_suffix}.intsct.vcf.gz")

    if cmp_intervals is not None:
        vcf_pipeline_utils.intersect_with_intervals(reheader_fn, cmp_intervals, select_intervals_fn)
    else:
        shutil.copy(reheader_fn, select_intervals_fn)
        vcf_pipeline_utils.index_vcf(select_intervals_fn)

    output_prefix = select_intervals_fn[:select_intervals_fn.index(".intsct.vcf.gz")]

    if concordance_tool == 'VCFEVAL':
        vcf_pipeline_utils.run_vcfeval_concordance(select_intervals_fn, truth_file, output_prefix,
                                                   ref_genome, cmp_intervals, call_sample, truth_sample, ignore_filter)
        output_prefix = f'{output_prefix}.vcfeval_concordance'
    else:
        vcf_pipeline_utils.run_genotype_concordance(select_intervals_fn, truth_file, output_prefix,
                                                    cmp_intervals, call_sample, truth_sample, ignore_filter)
        output_prefix = f'{output_prefix}.genotype_concordance'

    vcf_pipeline_utils.filter_bad_areas(select_intervals_fn, highconf_intervals, runs_intervals)
    vcf_pipeline_utils.filter_bad_areas(output_prefix + ".vcf.gz",
                                        highconf_intervals, runs_intervals)
    if runs_intervals is not None:
        return select_intervals_fn.replace("vcf.gz", "runs.vcf.gz"), output_prefix + ".runs.vcf.gz"
    else:
        return select_intervals_fn.replace("vcf.gz", "highconf.vcf.gz"), output_prefix + ".highconf.vcf.gz"
