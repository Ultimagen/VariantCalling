import subprocess
import gzip 
import pandas as pd
import pysam
import numpy as np 
import tqdm 
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

    input_files = [ f'{input_prefix}.{x}.vcf' for x in range(1,n_parts+1)]
    cmd = ['bcftools', 'concat', '-o', output_fname, '-O', 'z'] + input_files
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bcftools','index','-t', output_fname]
    subprocess.check_call(cmd)

def reheader_vcf( input_file: str, new_header: str, output_file: str) : 
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

    cmd = ['bcftools','reheader', '-h',new_header, input_file]
    with open(output_file, "wb") as out : 
        subprocess.check_call(cmd, stdout=out)
    cmd =['bcftools', 'index','-t',output_file]
    subprocess.check_call(cmd)


def run_genotype_concordance ( input_file: str, truth_file: str, output_prefix: str, 
    comparison_intervals: str,input_sample: str='NA12878', truth_sample='HG001'): 
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

    cmd = ['picard','GenotypeConcordance', 'CALL_VCF={}'.format(input_file), \
            'CALL_SAMPLE={}'.format(input_sample), 'O={}'.format(output_prefix), \
            'TRUTH_VCF={}'.format(truth_file), 'INTERVALS={}'.format(comparison_intervals), \
            'TRUTH_SAMPLE={}'.format(truth_sample), 'OUTPUT_VCF=true']
    subprocess.check_call(cmd)

    cmd = ['gunzip','-f', f'{output_prefix}.genotype_concordance.vcf.gz']
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    with open(f'{output_prefix}.genotype_concordance.vcf') as input_file : 
        with open(f'{output_prefix}.genotype_concordance.tmp','w') as output_file : 
            for line in input_file : 
                if line.startswith("##FORMAT=<ID=PS"):
                    print("Replacing")
                    output_file.write(line.replace("Type=Integer", "Type=String"))
                else:
                    output_file.write(line)
    cmd = ['mv', output_file.name, input_file.name]
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bgzip',input_file.name]
    subprocess.check_call(cmd)
    cmd = ['bcftools','index', '-tf', f'{input_file.name}.gz']
    subprocess.check_call(cmd)


def filter_bad_areas(input_file_calls: str, highconf_regions: str, runs_regions: str ):
    '''Looks at concordance only around high confidence areas and not around runs

    Parameters
    ----------
    input_file_calls: str
        Calls file
    highconf_regions: str
        High confidence regions bed
    runs_regions: str
        Runs
    '''

    highconf_file_name = input_file_calls.replace("vcf.gz","highconf.vcf")
    runs_file_name = input_file_calls.replace("vcf.gz","runs.vcf")
    with open(highconf_file_name,"wb") as highconf_file : 
        cmd = ['bedtools', 'intersect', '-a', input_file_calls, '-b', highconf_regions,'-nonamecheck',
            '-wa', '-header']
        subprocess.check_call(cmd, stdout=highconf_file)
    with open(runs_file_name, "wb") as runs_file : 
        cmd = ['bedtools', 'subtract', '-a', highconf_file_name, '-b', runs_regions,'-nonamecheck',
            '-A', '-header']
        subprocess.check_call(cmd, stdout=runs_file)

    cmd = ['bgzip','-f', highconf_file_name]
    subprocess.check_call(cmd)
    highconf_file_name += '.gz'
    cmd = ['bcftools','index','-tf', highconf_file_name]
    subprocess.check_call(cmd)
    cmd = ['bgzip','-f', runs_file_name]
    subprocess.check_call(cmd)
    runs_file_name += '.gz'
    cmd = ['bcftools','index','-tf', runs_file_name]
    subprocess.check_call(cmd)


def vcf2concordance(raw_calls_file: str, concordance_file: str) -> pd.DataFrame: 
    '''Generates concordance dataframe

    Parameters
    ----------
    raw_calls_file :str
        File with GATK calls
    concordance_file: str
        GenotypeConcordance file

    Returns
    -------
    pd.DataFrame
    '''

    vf = pysam.VariantFile(concordance_file)
    concordance = [ ( x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[0]['GT'], x.samples[1]['GT']) for x in vf]

    concordance = pd.DataFrame(concordance)
    concordance.columns = ['chrom','pos','qual','ref','alleles', 'gt_ultima', 'gt_ground_truth']
    concordance['indel'] = concordance['alleles'].apply(lambda x: len(set(([len(y) for y in x])))>1)
    def classify(x):
        if x['gt_ultima']==(None, None):
            return 'fn'
        elif x['gt_ground_truth']==(None, None):
            return 'fp'
        else:
            return 'tp'
    concordance['classify'] = concordance.apply(classify, axis=1)
    concordance.index = [(x[1]['chrom'],x[1]['pos']) for x in concordance.iterrows()]
    vf = pysam.VariantFile(raw_calls_file)
    original =pd.DataFrame( [ ( x.chrom, x.pos, x.qual, x.info['SOR']) for x in vf])
    original.columns = ['chrom','pos','qual','sor']
    original.index = [(x[1]['chrom'],x[1]['pos']) for x in original.iterrows()]
    original.drop('qual',axis=1,inplace=True)
    concordance = concordance.join(original.drop(['chrom','pos'], axis=1))
    return concordance    

def find_thresholds( concordance: pd.DataFrame ) -> pd.DataFrame : 
    quals = np.linspace(0,2000,30)
    sors = np.linspace(0,10,40)
    results = []
    pairs = []
    for q in tqdm.tqdm_notebook(quals) : 
        for s in sors : 
            pairs.append((q,s))
            tmp = (concordance[((concordance['qual']>q)&(concordance['sor']<s)) |\
                               (concordance['classify']=='fn')][['classify', 'indel']]).copy()
            tmp1 = (concordance[((concordance['qual']<q)|(concordance['sor']>s)) &\
                               (concordance['classify']=='tp')][['classify', 'indel']]).copy()
            tmp1['classify']='fn'
            tmp2=pd.concat((tmp, tmp1))
            results.append(tmp2.groupby(['classify','indel']).size())
    results = pd.concat(results,axis=1)
    results = results.T
    results.columns = results.columns.to_flat_index()

    results[('recall', 'indel')] = results[('tp',True)]/(results[('tp',True)]+results[('fn',True)])
    results[('specificity', 'indel')] = results[('tp',True)]/(results[('tp',True)]+results[('fp',True)])
    results[('recall', 'snp')] = results[('tp',False)]/(results[('tp',False)]+results[('fn',False)])
    results[('specificity', 'snp')] = results[('tp',False)]/(results[('tp',False)]+results[('fp',False)])
    results.index=pairs
    return results

def annotate_concordance(df: pd.DataFrame, fasta: str, alnfile: str) -> pd.DataFrame: 
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
    df = vcftools.get_coverage( df, alnfile, 10 )
    return df
