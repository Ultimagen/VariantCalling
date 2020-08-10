import subprocess
import pandas as pd
import pysam
import os.path
from collections import defaultdict
import python.vcftools as vcftools
from typing import Optional, List


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
    cmd = ['gatk', 'SelectVariants', '-V', input_fn, '-L', intervals_fn, '-O', output_fn]
    subprocess.check_call(cmd)


def run_genotype_concordance(input_file: str, truth_file: str, output_prefix: str,
                             comparison_intervals: str,
                             input_sample: str='NA12878', truth_sample='HG001',
                             ignore_filter: bool=False):
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
    ignore_filter: bool
        Ignore status of the variant filter
    Returns
    -------
    None
    '''

    cmd = ['picard', 'GenotypeConcordance', 'CALL_VCF={}'.format(input_file),
           'CALL_SAMPLE={}'.format(input_sample), 'O={}'.format(output_prefix),
           'TRUTH_VCF={}'.format(truth_file), 'INTERVALS={}'.format(
               comparison_intervals),
           'TRUTH_SAMPLE={}'.format(truth_sample), 'OUTPUT_VCF=true',
           'IGNORE_FILTER_STATUS={}'.format(ignore_filter)]
    subprocess.check_call(cmd)

    cmd = ['gunzip', '-f', f'{output_prefix}.genotype_concordance.vcf.gz']
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    with open(f'{output_prefix}.genotype_concordance.vcf') as input_file_handle:
        with open(f'{output_prefix}.genotype_concordance.tmp', 'w') as output_file_handle:
            for line in input_file_handle:
                if line.startswith("##FORMAT=<ID=PS"):
                    output_file_handle.write(line.replace(
                        "Type=Integer", "Type=String"))
                else:
                    output_file_handle.write(line)
    cmd = ['mv', output_file_handle.name, input_file_handle.name]
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bgzip', input_file_handle.name]
    subprocess.check_call(cmd)
    cmd = ['bcftools', 'index', '-tf', f'{input_file_handle.name}.gz']
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
            if len(set_gtr & set_ultima) > 0:
                return 'tp'
            elif len(set_ultima - set_gtr) > 0:
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

    concordance_df.loc[(concordance_df['classify_gt'] == 'tp') & (concordance_df['classify'] == 'fp'),'classify_gt'] = 'fp'

    concordance_df.index = [(x[1]['chrom'], x[1]['pos'])
                            for x in concordance_df.iterrows()]
    vf = pysam.VariantFile(raw_calls_file)
    vfi = map(lambda x: defaultdict(lambda: None, x.info.items() +
                                    x.samples[0].items() + [('QUAL', x.qual), ('CHROM', x.chrom), ('POS', x.pos),
                                                            ('FILTER', ';'.join(x.filter.keys()))]), vf)
    columns = ['chrom', 'pos', 'filter', 'qual', 'sor', 'as_sor',
               'as_sorp', 'fs', 'vqsr_val', 'qd', 'dp', 'ad', 'tree_score']
    original = pd.DataFrame([[x[y.upper()] for y in columns] for x in vfi])
    original.columns = columns
    original.index = [(x[1]['chrom'], x[1]['pos'])
                      for x in original.iterrows()]
    if format != 'VCFEVAL':
        original.drop('qual', axis=1, inplace=True)
    else:
        concordance_df.drop('qual', axis=1, inplace=True)
    concordance = concordance_df.join(original.drop(['chrom', 'pos'], axis=1))
    only_ref = concordance.alleles.apply(len)==1
    concordance = concordance[~only_ref]
    return concordance


def annotate_concordance(df: pd.DataFrame, fasta: str,
                         alnfile: Optional[str] = None,
                         annotate_intervals: List[str] = [],
                         runfile: Optional[str] = None, 
                         flow_order: Optional[str] = "TACG", 
                         hmer_run_length_dist: Optional[tuple] = (10,10)) -> pd.DataFrame:
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

    df = vcftools.classify_indel(df)
    df = vcftools.is_hmer_indel(df, fasta)
    df = vcftools.get_motif_around(df, 5, fasta)
    df = vcftools.get_gc_content(df, 10, fasta)
    if alnfile is not None:
        df = vcftools.get_coverage(df, alnfile, 10)
    if runfile is not None:
        length, dist = hmer_run_length_dist
        df = vcftools.close_to_hmer_run(
            df, runfile, min_hmer_run_length=length, max_distance=dist)
    if annotate_intervals is not None:
        for annotation_file in annotate_intervals:
            df = vcftools.annotate_intervals(df, annotation_file)
    df = vcftools.fill_filter_column(df)
    df = vcftools.annotate_cycle_skip(df, flow_order="TACG")
    return df



class FilterWrapper:
        def __init__(self, df: pd.DataFrame):
            self.orig_df = df
            self.df = df
            self.reset()

        def reset(self):
            self.df = self.orig_df
            return self

        # here we also keep tp which are low_score.
        # We consider them also as fn
        def get_fn(self):
            if 'filter' in self.df.columns:
                self.df = self.df[
                    (self.df['classify'] == 'fn') | ((self.df['classify'] == 'tp') & self.filtering(self.df['filter']))]
            else:
                self.df = self.df[
                    (self.df['classify'] == 'fn') | (self.df['classify'] == 'tp')]
            return self

        def get_fp(self):
            self.df = self.df[self.df['classify'] == 'fp']
            return self

        def get_fp_diff(self):
            self.df = self.df[(self.df['classify'] == 'tp') & (self.df['classify_gt'] == 'fp')]
            return self

        def get_fn_diff(self):
            self.df = self.df[((self.df['classify'] == 'tp') & (self.df['classify_gt'] == 'fn'))]
            return self

        def get_SNP(self):
            self.df = self.df[self.df['indel'] == False]
            return self

        def get_h_mer(self, val_start:int =1, val_end:int =999):
            self.df = self.df[(self.df['hmer_indel_length'] >= val_start) & (self.df['indel'] == True)]
            self.df = self.df[(self.df['hmer_indel_length'] <= val_end)]
            return self

        def get_non_h_mer(self):
            self.df = self.df[(self.df['hmer_indel_length'] == 0) & (self.df['indel'] == True)]
            return self

        def get_df(self):
            return self.df

        def filtering(self,filter_column):
            return ~filter_column.str.contains('LOW_SCORE', regex=False)

        # converts the h5 format to the BED format
        def BED_format(self):
            do_filtering = 'filter' in self.df.columns
            if do_filtering:
                filter_column = self.df['filter']

            hmer_length_column = self.df['hmer_indel_length']
            # end pos
            # we want to add the rgb column, so we need to add all the columns before it
            self.df = pd.concat([self.df['chrom'],  # chrom
                                 self.df['pos'] - 1,  # chromStart
                                 self.df['pos'],  # chromEnd
                                 hmer_length_column], axis=1)  # name

            self.df.columns = ['chrom', 'chromStart', 'chromEnd', 'name']

            # decide a color by filter column
            if do_filtering:
                rgb_color = self.filtering(filter_column)
                rgb_color[rgb_color] = "0,0,255"  # blue
                rgb_color[rgb_color == False] = "121,121,121"  # grey
                self.df['score'] = 500
                self.df['strand'] = "."
                self.df['thickStart'] = self.df['chromStart']
                self.df['thickEnd'] = self.df['chromEnd']
                self.df['itemRgb'] = rgb_color
                self.df.columns = ['chrom', 'chromStart', 'chromEnd', 'name',
                                   'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
            return self

def bed_files_output(data: pd.DataFrame, output_file: str) -> None:
    '''Create a set of bed file tracks that are often used in the
    debugging and the evaluation of the variant calling results

    Parameters
    ----------
    df : pd.DataFrame
        Concordance dataframe

    Returns
    -------
    None
    '''

    basename, file_extension = os.path.splitext(output_file)

    # SNP filtering
    # fp
    snp_fp = FilterWrapper(data).get_SNP().get_fp().BED_format().get_df()
    # fn
    snp_fn = FilterWrapper(data).get_SNP().get_fn().BED_format().get_df()

    # Diff filtering
    # fp
    all_fp_diff = FilterWrapper(data).get_fp_diff().BED_format().get_df()
    # fn
    all_fn_diff = FilterWrapper(data).get_fn_diff().BED_format().get_df()

    # Hmer filtering
    # 1 to 3
    # fp
    hmer_fp_1_3 = FilterWrapper(data).get_h_mer(val_start=1, val_end=3).get_fp().BED_format().get_df()
    # fn
    hmer_fn_1_3 = FilterWrapper(data).get_h_mer(val_start=1, val_end=3).get_fn().BED_format().get_df()

    # 4 until 7
    # fp
    hmer_fp_4_7 = FilterWrapper(data).get_h_mer(val_start=4, val_end=7).get_fp().BED_format().get_df()
    # fn
    hmer_fn_4_7 = FilterWrapper(data).get_h_mer(val_start=4, val_end=7).get_fn().BED_format(
        ).get_df()

    # 18 and more
    # fp
    hmer_fp_8_end = FilterWrapper(data).get_h_mer(val_start=8).get_fp().BED_format().get_df()
    # fn
    hmer_fn_8_end = FilterWrapper(data).get_h_mer(val_start=8).get_fn().BED_format().get_df()

    # non-Hmer filtering
    # fp
    non_hmer_fp = FilterWrapper(data).get_non_h_mer().get_fp().BED_format().get_df()
    # fn
    non_hmer_fn = FilterWrapper(data).get_non_h_mer().get_fn().BED_format().get_df()

    def save_bed_file(file: pd.DataFrame, basename: str, curr_name: str) -> None:
        file.to_csv((basename + "_" + f"{curr_name}.bed"), sep='\t', index=False, header=False)

    save_bed_file(snp_fp, basename, "snp_fp")
    save_bed_file(snp_fn, basename, "snp_fn")

    save_bed_file(all_fp_diff, basename, "genotyping_errors_fp")
    save_bed_file(all_fn_diff, basename, "genotyping_errors_fn")

    save_bed_file(hmer_fp_1_3, basename, "hmer_fp_1_3")
    save_bed_file(hmer_fn_1_3, basename, "hmer_fn_1_3")
    save_bed_file(hmer_fp_4_7, basename, "hmer_fp_4_7")
    save_bed_file(hmer_fn_4_7, basename, "hmer_fn_4_7")
    save_bed_file(hmer_fp_8_end, basename, "hmer_fp_8_end")
    save_bed_file(hmer_fn_8_end, basename, "hmer_fn_8_end")

    save_bed_file(non_hmer_fp, basename, "non_hmer_fp")
    save_bed_file(non_hmer_fn, basename, "non_hmer_fn")
