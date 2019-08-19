import pysam
import pandas as pd
import pyfaidx
import tqdm
from . import utils


def get_concordance(genotype_concordance_vcf: str,
                    input_vcf: str,
                    reference_genome: str) -> pd.DataFrame:
    '''Generate concordance dataframe 

    Parameters
    ----------
    genotype_concordance_vcf: str
        Output of genotypeConcordance. Note - fix the PS format in the header. It should read String
    input_vcf: str
        Call input VCF (Calls VCF into GenotypeConcordance)
    reference_genome: str
        Reference genome FASTA (FAIDX-d) 
    Returns
    -------
    pd.DataFrame: 
        Concordance dataframe. The following columns will be added to the dataframe:
        * chrom - chromosome
        * pos - position
        * qual - QUAL field 
        * ref
        * alleles
        * gt_calls - call file genotype
        * gt_ground_truth - ground truth file genotype
        * indel - is the variant indel
        * is_hp - is the indel homopolymer deletion/insertion
        * classify - is the variant fp, tp, or fn ?
        * sor - SOR field
        * the dataframe will be indexed by tuple (chrom, pos)

    '''
    vf = pysam.VariantFile(genotype_concordance_vcf)

    concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[
                    0]['GT'], x.samples[1]['GT']) for x in vf]

    concordance = pd.DataFrame(concordance)
    concordance.columns = ['chrom', 'pos', 'qual',
                           'ref', 'alleles', 'gt_calls', 'gt_ground_truth']

    concordance['indel'] = concordance['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    def classify(x):
        if x['gt_calls'] == (None, None):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None):
            return 'fp'
        else:
            return 'tp'

    concordance['classify'] = concordance.apply(classify, axis=1)
    concordance.index = [(x[1]['chrom'], x[1]['pos'])
                         for x in concordance.iterrows()]
    vf = pysam.VariantFile(input_vcf)
    original = pd.DataFrame(
        [(x.chrom, x.pos, x.info['SOR']) for x in vf])
    original.columns = ['chrom', 'pos','sor']
    original.index = [(x[1]['chrom'], x[1]['pos'])
                      for x in original.iterrows()]
    concordance = concordance.join(original.drop(['chrom', 'pos'], axis=1))
    return concordance

def get_vcf_df( variant_calls : str) -> pd.DataFrame:
    '''Reads VCF file into dataframe

    Parameters
    ----------
    variant_calls: str
        VCF file

    Returns
    -------
    pd.DataFrame
    '''
    vf = pysam.VariantFile(variant_calls)

    concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[0]['GT']) for x in tqdm.tqdm_notebook(vf)]

    concordance = pd.DataFrame(concordance)
    concordance.columns = ['chrom', 'pos', 'qual',
                           'ref', 'alleles', 'gt']

    concordance['indel'] = concordance['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    concordance.index = [(x[1]['chrom'], x[1]['pos'])
                         for x in concordance.iterrows()]
    return concordance


def summarize_concordance(concordance: pd.DataFrame):
    '''Summarize variant concordance

    Parameters
    ----------
    concordance: pd.DataFrame
        Output of `get_concordance`

    Returns
    -------
    pd.DataFrame
        Summary dataframe
    '''
    tp_snp = ((~concordance['indel']) & (
        concordance['classify'] == 'tp')).sum()
    fp_snp = ((~concordance['indel']) & (
        concordance['classify'] == 'fp')).sum()
    fn_snp = ((~concordance['indel']) & (
        concordance['classify'] == 'fn')).sum()
    recall_snp = tp_snp / (tp_snp + fn_snp)
    precision_snp = tp_snp / (tp_snp + fp_snp)
    tp_indel = ((concordance['indel']) & (
        concordance['classify'] == 'tp')).sum()
    fp_indel = ((concordance['indel']) & (
        concordance['classify'] == 'fp')).sum()
    fn_indel = ((concordance['indel']) & (
        concordance['classify'] == 'fn')).sum()
    recall_indel = tp_indel / (tp_indel + fn_indel)
    precision_indel = tp_indel / (tp_indel + fp_indel)
    return pd.Series({"tp_snp": tp_snp,
                      "fp_snp": fp_snp,
                      "fn_snp": fn_snp,
                      "tp_indel": tp_indel,
                      "fp_indel": fp_indel,
                      "fn_indel": fn_indel,
                      "recall_snp": recall_snp,
                      "precision_snp": precision_snp,
                      "recall_indel": recall_indel,
                      "precision_indel": precision_indel})


def classify_indel(concordance: pd.DataFrame) -> pd.DataFrame:
    '''Classifies indel as insertion or deletion

    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe of concordances

    Returns
    -------
    pd.DataFrame: 
        Modifies dataframe by adding columns "indel_classify" and "indel_length"
    '''

    def classify(x):
        if not x['indel']:
            return None
        elif len(x['ref']) < max([len(y) for y in x['alleles']]):
            return 'ins'
        return 'del'
    concordance['indel_classify'] = concordance.apply(classify, axis=1)
    concordance['indel_length'] = concordance.apply(lambda x: abs( len(x['ref']) - max([len(y) for y in x['alleles'] if y != x['ref']])), axis=1)
    return concordance


def is_hmer_indel(concordance: pd.DataFrame, fasta_file: str) -> pd.DataFrame:
    '''
    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe of concordance or of VariantCalls. Should contain a collumn "indel_classify"
    fasta_file: str
        FAI indexed fasta file 

    Returns
    -------
    pd.DataFrame
        Adds column hmer_indel_length, hmer_indel_nuc
    '''

    fasta_idx = pyfaidx.Fasta(fasta_file)

    def _is_hmer(rec, fasta_idx):
        if not rec['indel']:
            return (0, None)

        if rec['indel_classify'] == 'ins':
            alt = [x for x in rec['alleles'] if x != rec['ref']][0][1:]
            if len(set(alt)) != 1:
                return (0, None)
            elif fasta_idx[rec['chrom']][rec['pos']].seq.upper() != alt[0]:
                
                return (0, None)
            else:
                return (len(alt) + utils.hmer_length(fasta_idx[rec['chrom']], rec['pos']), alt[0])

        if rec['indel_classify'] == 'del':
            del_seq = rec['ref'][1:]
            if len(set(del_seq)) != 1:
                return (0, None)
            elif fasta_idx[rec['chrom']][rec['pos'] + len(rec['ref'])-1].seq.upper() != del_seq[0]:
                return (0, None)
            else:
                return (len(del_seq) + utils.hmer_length(fasta_idx[rec['chrom']], rec['pos'] + len(rec['ref'])-1), del_seq[0])

    results = concordance.apply(lambda x: _is_hmer(x, fasta_idx), axis=1)
    concordance['hmer_indel_length'] = [x[0] for x in results]
    concordance['hmer_indel_nuc'] = [ x[1] for x in results ]
    return concordance