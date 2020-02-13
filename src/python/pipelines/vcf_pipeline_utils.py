import subprocess
import pandas as pd
import pysam
from os.path import exists
from collections import defaultdict
import python.vcftools as vcftools

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
    input_files = [x for x in input_files if exists(x)]
    cmd = ['bcftools', 'concat', '-o', output_fname, '-O', 'z'] + input_files
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bcftools', 'index', '-t', output_fname]
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
    with open(output_file, "wb") as out:
        subprocess.check_call(cmd, stdout=out)
    cmd = ['bcftools', 'index', '-t', output_file]
    subprocess.check_call(cmd)


def run_genotype_concordance(input_file: str, truth_file: str, output_prefix: str,
                             comparison_intervals: str, input_sample: str='NA12878', truth_sample='HG001'):
    '''Run GenotypeConcordance, correct the bug and reindex

    Parameters
    ----------
    input_file: str
        Our variant calls
    truth_file: str
        GIAB (or other source truth file)
    output_prefix: str
        Output prefix
    comparison_intervals: str
        Picard intervals file to make the comparisons on
    input_sample: str
        Name of the sample in our input_file
    truth_samle: str
        Name of the sample in the truth file

    Returns
    -------
    None
    '''

    cmd = ['picard', 'GenotypeConcordance', 'CALL_VCF={}'.format(input_file),
           'CALL_SAMPLE={}'.format(input_sample), 'O={}'.format(output_prefix),
           'TRUTH_VCF={}'.format(truth_file), 'INTERVALS={}'.format(
               comparison_intervals),
           'TRUTH_SAMPLE={}'.format(truth_sample), 'OUTPUT_VCF=true']
    subprocess.check_call(cmd)

    cmd = ['gunzip', '-f', f'{output_prefix}.genotype_concordance.vcf.gz']
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    with open(f'{output_prefix}.genotype_concordance.vcf') as input_file:
        with open(f'{output_prefix}.genotype_concordance.tmp', 'w') as output_file:
            for line in input_file:
                if line.startswith("##FORMAT=<ID=PS"):
                    print("Replacing")
                    output_file.write(line.replace(
                        "Type=Integer", "Type=String"))
                else:
                    output_file.write(line)
    cmd = ['mv', output_file.name, input_file.name]
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bgzip', input_file.name]
    subprocess.check_call(cmd)
    cmd = ['bcftools', 'index', '-tf', f'{input_file.name}.gz']
    subprocess.check_call(cmd)


def filter_bad_areas(input_file_calls: str, highconf_regions: str, runs_regions: str):
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
               '-wa', '-header']
        subprocess.check_call(cmd, stdout=highconf_file)

    cmd = ['bgzip', '-f', highconf_file_name]
    subprocess.check_call(cmd)
    highconf_file_name += '.gz'
    cmd = ['bcftools', 'index', '-tf', highconf_file_name]
    subprocess.check_call(cmd)

    if runs_regions is not None:
        with open(runs_file_name, "wb") as runs_file:
            cmd = ['bedtools', 'subtract', '-a', highconf_file_name, '-b', runs_regions, '-nonamecheck',
                   '-A', '-header']
            subprocess.check_call(cmd, stdout=runs_file)

        cmd = ['bgzip', '-f', runs_file_name]
        subprocess.check_call(cmd)
        runs_file_name += '.gz'
        cmd = ['bcftools', 'index', '-tf', runs_file_name]
        subprocess.check_call(cmd)


def vcf2concordance(raw_calls_file: str, concordance_file: str, format: str = 'GC') -> pd.DataFrame:
    '''Generates concordance dataframe

    Parameters
    ----------
    raw_calls_file :str
        File with GATK calls
    concordance_file: str
        GenotypeConcordance file
    format: str
        Either 'GC' or 'VCFEVAL' - format for the concordance_file

    Returns
    -------
    pd.DataFrame
    '''

    vf = pysam.VariantFile(concordance_file)
    if format == 'GC':
        concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[
                        0]['GT'], x.samples[1]['GT']) for x in vf]
    elif format == 'VCFEVAL':
        concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles,
                        x.samples[1]['GT'], x.samples[0]['GT']) for x in vf if 'CALL' not in x.info.keys() or
                       x.info['CALL'] != 'OUT']

    concordance_df: pd.DataFrame = pd.DataFrame(concordance)
    concordance_df.columns = ['chrom', 'pos', 'qual',
                              'ref', 'alleles', 'gt_ultima', 'gt_ground_truth']
    concordance_df['indel'] = concordance_df['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    def classify(x):
        if x['gt_ultima'] == (None, None) or x['gt_ultima'] == (None,):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None) or x['gt_ground_truth'] == (None,):
            return 'fp'
        else:
            set_gtr = set(x['gt_ground_truth']) - set([0])
            set_ultima = set(x['gt_ultima']) - set([0])
            if set_gtr == set_ultima:
                return 'tp'
            elif set_ultima - set_gtr:
                return 'fp'
            else:
                return 'fn'

    concordance_df['classify'] = concordance_df.apply(classify, axis=1)

    def classify_gt(x):
        if x['gt_ultima'] == (None, None) or x['gt_ultima'] == (None,):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None) or x['gt_ground_truth'] == (None,):
            return 'fp'
        elif (x['gt_ultima'] == (0, 1) or x['gt_ultima'] == (1, 0)) and x['gt_ground_truth'] == (1, 1):
            return 'fn'
        elif (x['gt_ground_truth'] == (0, 1) or x['gt_ground_truth'] == (1, 0)) and x['gt_ultima'] == (1, 1):
            return 'fp'
        else:
            return 'tp'
    concordance_df['classify_gt'] = concordance_df.apply(classify_gt, axis=1)

    concordance_df.index = [(x[1]['chrom'], x[1]['pos'])
                            for x in concordance_df.iterrows()]
    vf = pysam.VariantFile(raw_calls_file)
    vfi = map(lambda x: defaultdict(lambda: None, x.info.items(
    ) + x.samples[0].items() + [('QUAL', x.qual), ('CHROM', x.chrom), ('POS', x.pos)]), vf)
    columns = ['chrom', 'pos', 'qual', 'sor', 'as_sor',
               'as_sorp', 'fs', 'vqsr_val', 'qd', 'dp', 'ad']
    original = pd.DataFrame([[x[y.upper()] for y in columns] for x in vfi])
    original.columns = columns
    original.index = [(x[1]['chrom'], x[1]['pos'])
                      for x in original.iterrows()]
    if format != 'VCFEVAL':
        original.drop('qual', axis=1, inplace=True)
    else:
        concordance_df.drop('qual', axis=1, inplace=True)
    concordance = concordance_df.join(original.drop(['chrom', 'pos'], axis=1))
    return concordance


def annotate_concordance(df: pd.DataFrame, fasta: str, alnfile: str, runfile: str) -> pd.DataFrame:
    '''Annotates concordance data with information about SNP/INDELs and motifs

    Parameters
    ----------
    df: pd.DataFrame
        Concordance dataframe
    fasta: str
        Indexed FASTA of the reference genome
    alnfile: str
        Alignment file
    '''

    df = vcftools.classify_indel(df)
    df = vcftools.is_hmer_indel(df, fasta)
    df = vcftools.get_motif_around(df, 5, fasta)
    if alnfile is not None:
        df = vcftools.get_coverage(df, alnfile, 10)
    df = vcftools.close_to_hmer_run(
        df, runfile, min_hmer_run_length=10, max_distance=10)
    return df
