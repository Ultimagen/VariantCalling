import subprocess
import numpy as np
import pandas as pd
import pysam
import pyfaidx
import os.path
import shutil
from collections import defaultdict
import python.vcftools as vcftools
import python.modules.variant_annotation as annotation
import python.modules.flow_based_concordance as fbc
from typing import Optional, List
import logging
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)


def combine_vcf(n_parts: int, input_prefix: str, output_fname: str):
    '''Combines VCF in parts from GATK and indices the result
    Parameters
    ----------
    n_parts: int
        Number of VCF parts (names will be 1-based)
    input_prefix: str
        Prefix of the VCF files (including directory) 1.vcf.gz ... will be added
    output_fname: str
        Name of the output VCF
    '''
    input_files = [f'{input_prefix}.{x}.vcf' for x in range(1, n_parts + 1)] +\
        [f'{input_prefix}.{x}.vcf.gz' for x in range(1, n_parts + 1)]
    input_files = [x for x in input_files if os.path.exists(x)]
    cmd = ['bcftools', 'concat', '-o', output_fname, '-O', 'z'] + input_files
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    index_vcf(output_fname)


def index_vcf(vcf: str):
    '''Tabix index on VCF'''
    cmd = ['bcftools', 'index', '-tf', vcf]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)


def reheader_vcf(input_file: str, new_header: str, output_file: str):
    '''Run bcftools reheader and index

    Parameters
    ----------
    input_file: str
        Input file name
    new_header: str
        Name of the new header
    output_file: str
        Name of the output file

    Returns
    -------
    None, generates `output_file`
    '''

    cmd = ['bcftools', 'reheader', '-h', new_header, input_file]
    logger.info(" ".join(cmd))
    with open(output_file, "wb") as out:
        subprocess.check_call(cmd, stdout=out)
    index_vcf(output_file)

class IntervalFile:
    def __init__(self,cmp_intervals: str, ref: str, ref_dict: str):
        # determine the file type and create the other temporary copy
        if cmp_intervals is None:
            self._is_none = True
            self._interval_list_file_name = None
            self._bed_file_name = None

        elif cmp_intervals.endswith('.interval_list'):
            self._interval_list_file_name = cmp_intervals
            # create the interval bed file
            cmd = ["picard", "IntervalListToBed",
                   f"I={cmp_intervals}",
                   f"O={os.path.splitext(cmp_intervals)[0]}.bed"]
            logger.info(" ".join(cmd))
            subprocess.check_call(cmd)
            self._bed_file_name = f"{os.path.splitext(cmp_intervals)[0]}.bed"
            self._is_none = False

        elif cmp_intervals.endswith('.bed'):
            self._bed_file_name = cmp_intervals
            # deduce ref_dict
            if ref_dict is None:
                ref_dict = f"{ref}.dict"
            if not os.path.isfile(ref_dict):
                logger.error(f"dict file does not exist: {ref_dict}")

            # create the interval list file
            cmd = ["picard", "BedToIntervalList",
                   f"I={cmp_intervals}",
                   f"O={os.path.splitext(cmp_intervals)[0]}.interval_list",
                   f"SD={ref_dict}"]
            logger.info(" ".join(cmd))
            subprocess.check_call(cmd)
            self._interval_list_file_name = f"{os.path.splitext(cmp_intervals)[0]}.interval_list"
            self._is_none = False
        else:
            logger.error("the cmp_intervals should be of type interval list or bed")
            self._is_none = True
            self._interval_list_file_name = None
            self._bed_file_name = None
    def as_bed_file(self):
        return self._bed_file_name
    def as_interval_list_file(self):
        return self._interval_list_file_name
    def is_none(self):
        return self._is_none

def intersect_bed_files(input_bed1: str, input_bed2: str, bed_output: str) -> None:
    '''Intersects bed files

    Parameters
    ----------
    input_bed1: str
        Input Bed file
    input_bed2: str
        Input Bed file
    bed_output: str
        Output bed intersected file

    Return
    ------
    None
        Writes output_fn file
    '''
    cmd = ['bedtools', 'intersect', '-a', input_bed1,
           '-b', input_bed2]
    logger.info(" ".join(cmd))
    with open(bed_output, "w") as f:
        subprocess.call(cmd, stdout=f)


def bed_file_length(input_bed: str) -> int:
    '''Calc the number of bases in a bed file

    Parameters
    ----------
    input_bed: str
        Input Bed file

    Return
    ------
    int
        number of bases in a bed file
    '''

    df = pd.read_csv(input_bed, sep="\t", header=None)
    df = df.iloc[:,[0,1,2]]
    df.columns = ['chr', 'pos_start', 'pos_end']
    return np.sum(df['pos_end']-df['pos_start']+1)

def intersect_with_intervals(input_fn: str, intervals_fn: str, output_fn: str) -> None:
    '''Intersects VCF with intervalList

    Parameters
    ----------
    input_fn: str
        Input file
            intervals_fn: str
        Interval_list filename
    output_fn: str
        Output file

    Return
    ------
    None
        Writes output_fn file
    '''
    cmd = ['gatk', 'SelectVariants', '-V', input_fn,
           '-L', intervals_fn, '-O', output_fn]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)


def run_genotype_concordance(input_file: str, truth_file: str, output_prefix: str,
                             comparison_intervals: Optional[str] = None,
                             input_sample: str = 'NA12878', truth_sample='HG001',
                             ignore_filter: bool = False):
    '''Run GenotypeConcordance, correct the bug and reindex

    Parameters
    ----------
    input_file: str
        Our variant calls
    truth_file: str
        GIAB (or other source truth file)
    output_prefix: str
        Output prefix
    comparison_intervals: Optional[str]
        Picard intervals file to make the comparisons on. Default (None - all genome)
    input_sample: str
        Name of the sample in our input_file
    truth_samle: str
        Name of the sample in the truth file
    ignore_filter: bool
        Ignore status of the variant filter
    Returns
    -------
    None
    '''

    cmd = ['picard', 'GenotypeConcordance', 'CALL_VCF={}'.format(input_file),
           'CALL_SAMPLE={}'.format(input_sample), 'O={}'.format(output_prefix),
           'TRUTH_VCF={}'.format(truth_file),
           'TRUTH_SAMPLE={}'.format(truth_sample), 'OUTPUT_VCF=true',
           'IGNORE_FILTER_STATUS={}'.format(ignore_filter)]
    if comparison_intervals is not None:
        cmd += ['INTERVALS={}'.format(comparison_intervals)]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    fix_vcf_format(f'{output_prefix}.genotype_concordance')


def run_vcfeval_concordance(input_file: str, truth_file: str, output_prefix: str,
                            ref_genome: str,
                            comparison_intervals: Optional[str] = None,
                            input_sample: str = 'NA12878', truth_sample='HG001',
                            ignore_filter: bool = False):
    '''Run vcfevalConcordance

    Parameters
    ----------
    input_file: str
        Our variant calls
    truth_file: str
        GIAB (or other source truth file)
    output_prefix: str
        Output prefix
    ref_genome: str
        Fasta reference file
    comparison_intervals: Optional[str]
        Picard intervals file to make the comparisons on. Default: None = all genome
    input_sample: str
        Name of the sample in our input_file
    truth_sample: str
        Name of the sample in the truth file
    ignore_filter: bool
        Ignore status of the variant filter
    Returns
    -------
    None
    '''

    output_dir = os.path.dirname(output_prefix)
    SDF_path = ref_genome + '.sdf'
    vcfeval_output_dir = os.path.join(
        output_dir, os.path.basename(output_prefix) + '.vcfeval_output')

    if os.path.isdir(vcfeval_output_dir):
        shutil.rmtree(vcfeval_output_dir)

    # filter the vcf to be only in the comparison_intervals.
    filtered_truth_file = os.path.join(output_dir, '.'.join(
        (os.path.basename(truth_file), 'filtered', 'vcf.gz')))
    if comparison_intervals is not None:
        intersect_with_intervals(
            truth_file, comparison_intervals, filtered_truth_file)
    else:
        shutil.copy(truth_file, filtered_truth_file)
        index_vcf(filtered_truth_file)

    # vcfeval calculation
    cmd = ['rtg', 'vcfeval',
           '-b', filtered_truth_file,
           '--calls', input_file,
           '-o', vcfeval_output_dir,
           '-t', SDF_path,
           '-m', 'combine',
           '--sample', f'{truth_sample},{input_sample}',
           '--decompose']
    if ignore_filter:
        cmd += ['--all-records']
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    # fix the vcf file format
    fix_vcf_format(os.path.join(vcfeval_output_dir, "output"))

    # make the vcfeval output file without weird variants
    cmd = ['bcftools', 'norm',
           '-f', ref_genome, '-m+any', '-o', os.path.join(
               vcfeval_output_dir, 'output.norm.vcf.gz'),
           '-O', 'z', os.path.join(vcfeval_output_dir, 'output.vcf.gz')
           ]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

    # move the file to be compatible with the output file of the genotype
    # concordance
    cmd = ['mv', os.path.join(vcfeval_output_dir, 'output.norm.vcf.gz'),
           output_prefix + '.vcfeval_concordance.vcf.gz']
    subprocess.check_call(cmd)

    # generate index file for the vcf.gz file
    index_vcf(output_prefix + '.vcfeval_concordance.vcf.gz')


def fix_vcf_format(output_prefix):
    cmd = ['gunzip', '-f', f'{output_prefix}.vcf.gz']
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    with open(f'{output_prefix}.vcf') as input_file_handle:
        with open(f'{output_prefix}.tmp', 'w') as output_file_handle:
            for line in input_file_handle:
                if line.startswith("##FORMAT=<ID=PS"):
                    output_file_handle.write(line.replace(
                        "Type=Integer", "Type=String"))
                else:
                    output_file_handle.write(line)
    cmd = ['mv', output_file_handle.name, input_file_handle.name]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bgzip', input_file_handle.name]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
    index_vcf(f'{input_file_handle.name}.gz')


def filter_bad_areas(input_file_calls: str, highconf_regions: str, runs_regions: Optional[str]):
    '''Looks at concordance only around high confidence areas and not around runs

    Parameters
    ----------
    input_file_calls: str
        Calls file
    highconf_regions: str
        High confidence regions bed
    runs_regions: str or None
        Runs
    '''

    highconf_file_name = input_file_calls.replace("vcf.gz", "highconf.vcf")
    runs_file_name = input_file_calls.replace("vcf.gz", "runs.vcf")
    with open(highconf_file_name, "wb") as highconf_file:
        cmd = ['bedtools', 'intersect', '-a', input_file_calls, '-b', highconf_regions, '-nonamecheck',
               '-header', '-u']
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd, stdout=highconf_file)

    cmd = ['bgzip', '-f', highconf_file_name]
    logger.info(" ".join(cmd))

    subprocess.check_call(cmd)
    highconf_file_name += '.gz'
    index_vcf(highconf_file_name)

    if runs_regions is not None:
        with open(runs_file_name, "wb") as runs_file:
            cmd = ['bedtools', 'subtract', '-a', highconf_file_name, '-b', runs_regions, '-nonamecheck',
                   '-A', '-header']
            logger.info(" ".join(cmd))
            subprocess.check_call(cmd, stdout=runs_file)

        cmd = ['bgzip', '-f', runs_file_name]
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd)
        runs_file_name += '.gz'
        index_vcf(runs_file_name)


def _fix_errors(df):
    # fix all the places in which vcfeval returns a good result, but the genotype is not adequate
    # in these cases we change the genotype of the gt to be adequate with the classify function as follow:
    # (TP,TP), (TP,None) - should put the values of ultima in the gt
    df.loc[(df['call'] == 'TP') & ((df['base'] == 'TP') | (df['base'].isna())), 'gt_ground_truth'] = \
        df[(df['call'] == 'TP') & ((df['base'] == 'TP')
                                   | (df['base'].isna()))]['gt_ultima']

    # (None, TP) (None,FN_CA) - remove these rows
    df.drop(df[(df['call'].isna()) & ((df['base'] == 'TP')
                                      | (df['base'] == 'FN_CA'))].index, inplace=True)

    # (FP_CA,FN_CA), (FP_CA,None) - Fake a genotype from ultima such that one of the alleles is the same (and only one)
    df.loc[(df['call'] == 'FP_CA') & ((df['base'] == 'FN_CA') | (df['base'].isna())), 'gt_ground_truth'] = \
        df[(df['call'] == 'FP_CA') & ((df['base'] == 'FN_CA') | (df['base'].isna()))]['gt_ultima']. \
        apply(lambda x: ((x[0], x[0]) if (x[1] == 0) else (
            (x[1], x[1]) if (x[0] == 0) else (x[0], 0))))
    return df


def vcf2concordance(raw_calls_file: str, concordance_file: str, format: str = 'GC', chromosome: str = None) -> pd.DataFrame:
    '''Generates concordance dataframe

    Parameters
    ----------
    raw_calls_file: str
        File with GATK calls (.vcf.gz)
    concordance_file: str
        GenotypeConcordance/VCFEVAL output file (.vcf.gz)
    format: str
        Either 'GC' or 'VCFEVAL' - format for the concordance_file
    chromosome: str
        Fetch a specific chromosome (Default - all)
    Returns
    -------
    pd.DataFrame
    '''

    if chromosome is None:
        vf = pysam.VariantFile(concordance_file)
    else:
        vf = pysam.VariantFile(concordance_file).fetch(chromosome)

    if format == 'GC':
        concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[
                        0]['GT'], x.samples[1]['GT']) for x in vf]
        column_names = ['chrom', 'pos', 'qual',
                        'ref', 'alleles', 'gt_ultima', 'gt_ground_truth']

    elif format == 'VCFEVAL':
        concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles,
                        x.samples[1]['GT'], x.samples[0]['GT'],
                        x.info.get('SYNC', None), x.info.get('CALL', None), x.info.get('BASE', None))
                       for x in vf if 'CALL' not in x.info.keys() or
                       ((x.info['CALL'] != 'OUT') and (x.info['CALL'] != 'IGN'))]
        column_names = ['chrom', 'pos', 'qual',
                        'ref', 'alleles', 'gt_ultima', 'gt_ground_truth', 'sync', 'call', 'base']

    concordance_df = pd.DataFrame(concordance, columns=column_names)
    if format == 'VCFEVAL':
        # make the gt_ground_truth compatible with GC
        concordance_df['gt_ground_truth'] =\
            concordance_df['gt_ground_truth'].map(
                lambda x: (None, None) if x == (None,) else x)

    concordance_df['indel'] = concordance_df['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    if format == 'VCFEVAL':
        concordance_df = _fix_errors(concordance_df)

    def classify(x):
        if x['gt_ultima'] == (None, None) or x['gt_ultima'] == (None,):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None) or x['gt_ground_truth'] == (None,):
            return 'fp'
        else:
            set_gtr = set(x['gt_ground_truth']) - set([0])
            set_ultima = set(x['gt_ultima']) - set([0])
            if len(set_gtr & set_ultima) > 0:
                return 'tp'
            elif len(set_ultima - set_gtr) > 0:
                return 'fp'
            else:
                return 'fn'
    concordance_df['classify'] = concordance_df.apply(
        classify, axis=1, result_type='reduce')

    def classify_gt(x):
        n_ref_gtr = len([y for y in x['gt_ground_truth'] if y == 0])
        n_ref_ultima = len([y for y in x['gt_ultima'] if y == 0])

        if x['gt_ultima'] == (None, None) or x['gt_ultima'] == (None,):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None) or x['gt_ground_truth'] == (None,):
            return 'fp'
        elif n_ref_gtr < n_ref_ultima:
            return 'fn'
        elif n_ref_gtr > n_ref_ultima:
            return 'fp'
        elif x['gt_ultima'] != x['gt_ground_truth']:
            return 'fp'
        else:
            return 'tp'
            
    concordance_df['classify_gt'] = concordance_df.apply(
        classify_gt, axis=1, result_type='reduce')

    concordance_df.loc[(concordance_df['classify_gt'] == 'tp') & (
        concordance_df['classify'] == 'fp'), 'classify_gt'] = 'fp'

    concordance_df.index = list(zip(concordance_df.chrom, concordance_df.pos))

    if chromosome is None:
        vf = pysam.VariantFile(raw_calls_file)
    else:
        vf = pysam.VariantFile(raw_calls_file).fetch(chromosome)
    vfi = map(lambda x: defaultdict(lambda: None, x.info.items() +
                                    x.samples[0].items() + [('QUAL', x.qual), ('CHROM', x.chrom), ('POS', x.pos),
                                                            ('FILTER', ';'.join(x.filter.keys()))]), vf)
    columns = ['chrom', 'pos', 'filter', 'qual', 'sor', 'as_sor',
               'as_sorp', 'fs', 'vqsr_val', 'qd', 'dp', 'ad',
               'tree_score', 'tlod', 'af','fpr','group', 'nlod','nalod']
    original = pd.DataFrame([[x[y.upper()] for y in columns]
                             for x in vfi], columns=columns)
    original.index = list(zip(original.chrom, original.pos))

    if format != 'VCFEVAL':
        original.drop('qual', axis=1, inplace=True)
    else:
        concordance_df.drop('qual', axis=1, inplace=True)
    concordance = concordance_df.join(original.drop(['chrom', 'pos'], axis=1))
    only_ref = concordance['alleles'].apply(len) == 1
    concordance = concordance[~only_ref]

    return concordance


def annotate_concordance(df: pd.DataFrame, fasta: str,
                         alnfile: Optional[str] = None,
                         annotate_intervals: List[str] = [],
                         runfile: Optional[str] = None,
                         flow_order: Optional[str] = "TACG",
                         hmer_run_length_dist: tuple = (10, 10)) -> pd.DataFrame:
    '''Annotates concordance data with information about SNP/INDELs and motifs

    Parameters
    ----------
    df : pd.DataFrame
        Concordance dataframe
    fasta : str
        Indexed FASTA of the reference genome
    alnfile : Optional[str], optional
        Alignment file (Optional)
    annotate_intervals : List[str], optional
        Description
    runfile : Optional[str], optional
        Description
    flow_order : Optional[str], optional
        Description

    Returns
    -------
    pd.DataFrame
        Annotated dataframe

    '''

    logger.info("Marking SNP/INDEL")
    df = annotation.classify_indel(df)
    logger.info("Marking H-INDEL")
    df = annotation.is_hmer_indel(df, fasta)
    logger.info("Maring motifs")
    df = annotation.get_motif_around(df, 5, fasta)
    logger.info("Marking GC content")
    df = annotation.get_gc_content(df, 10, fasta)
    if alnfile is not None:
        logger.info("Calculating coverage")
        df = annotation.get_coverage(df, alnfile, 10)
    if runfile is not None:
        length, dist = hmer_run_length_dist
        logger.info("Marking homopolymer runs")
        df = annotation.close_to_hmer_run(
            df, runfile, min_hmer_run_length=length, max_distance=dist)
    if annotate_intervals is not None:
        for annotation_file in annotate_intervals:
            logger.info("Annotating intervals")
            df = annotation.annotate_intervals(df, annotation_file)
    logger.debug("Filling filter column")  # debug since not interesting step
    df = annotation.fill_filter_column(df)

    logger.info("Filling filter column")
    df = annotation.annotate_cycle_skip(df, flow_order="TACG")
    return df


def reinterpret_variants(concordance_df: pd.DataFrame, reference_fasta: str,
                         ignore_low_quality_fps: bool = False) -> pd.DataFrame:
    '''Reinterprets the variants by comparing the variant to the ground truth in flow space

    Parameters
    ----------
    concordance_df: pd.DataFrame
        Input dataframe
    reference_fasta: str
        Indexed FASTA
    ignore_low_quality_fps: bool
        Shoud the low quality false positives be ignored in reinterpretation (True for mutect, default False)

    Returns
    -------
    pd.DataFrame
        Reinterpreted dataframe

    See Also
    --------
    `flow_based_concordance.py`
    '''
    logger.info("Variants reinterpret")
    concordance_df_result = pd.DataFrame()
    fasta = pyfaidx.Fasta(reference_fasta)
    for contig in concordance_df['chrom'].unique():
        concordance_df_contig = concordance_df.loc[
            concordance_df['chrom'] == contig]
        input_dict = _get_locations_to_work_on(
            concordance_df_contig, ignore_low_quality_fps)
        concordance_df_contig = fbc.reinterpret_variants(
            concordance_df_contig, input_dict, fasta)
        concordance_df_result = pd.concat(
            [concordance_df_result, concordance_df_contig])
    return concordance_df_result


def _get_locations_to_work_on(_df: pd.DataFrame, ignore_low_quality_fps: bool = False) -> dict:
    '''Dictionary of service locatoins

    Parameters
    ----------
    _df: pd.DataFrame
        Input
    ignore_low_quality_fps: bool
        Should we ignore the low quality false positives

    '''
    df = vcftools.FilterWrapper(_df)
    fps = df.reset().get_fp().get_df()
    if 'tree_score' in fps.columns and fps['tree_score'].dtype == np.float64 and ignore_low_quality_fps:
        cutoff = fps.tree_score.quantile(.80)
        fps = fps.query(f"tree_score > {cutoff}")
    fns = df.reset().get_df().query('classify=="fn"')
    tps = df.reset().get_tp().get_df()
    gtr = df.reset().get_df().loc[
        df.get_df()["gt_ground_truth"].apply(
            lambda x: x != (None, None) and x != (None,))
    ].copy()
    gtr.sort_values("pos", inplace=True)
    ugi = df.reset().get_df().loc[df.get_df()["gt_ultima"].apply(
        lambda x: x != (None, None) and x != (None,))].copy()
    ugi.sort_values("pos", inplace=True)

    pos_fps = np.array(fps.pos)
    pos_gtr = np.array(gtr.pos)
    pos_ugi = np.array(ugi.pos)
    pos_fns = np.array(fns.pos)

    result = {'fps': fps, 'fns': fns, 'tps': tps,
              'gtr': gtr, 'ugi': ugi, 'pos_fps': pos_fps,
              'pos_gtr': pos_gtr, 'pos_ugi': pos_ugi, 'pos_fns': pos_fns}

    return result
