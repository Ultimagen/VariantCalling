import pysam
import pandas as pd




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

    concordance = [(x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[0]['GT'], x.samples[1]['GT']) for x in vf]

    concordance = pd.DataFrame(concordance)
    concordance.columns = ['chrom', 'pos', 'qual', 'ref', 'alleles', 'gt_calls', 'gt_ground_truth']

    concordance['indel'] = concordance['alleles'].apply(lambda x: len(set(([len(y) for y in x]))) > 1)


    def classify(x):
        if x['gt_calls'] == (None, None):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None):
            return 'fp'
        else:
            return 'tp'


    concordance['classify'] = concordance.apply(classify, axis=1)
    concordance.index = [(x[1]['chrom'], x[1]['pos']) for x in concordance.iterrows()]
    vf = pysam.VariantFile(input_vcf)
    original = pd.DataFrame([(x.chrom, x.pos, x.qual, x.info['SOR']) for x in vf])
    original.columns = ['chrom', 'pos', 'sor']
    original.index = [(x[1]['chrom'], x[1]['pos']) for x in original.iterrows()]
    concordance = concordance.join(original.drop(['chrom', 'pos'], axis=1))
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
    tp_snp = ((~concordance['indel']) & (concordance['classify']=='tp')).sum()
    fp_snp = ((~concordance['indel']) & (concordance['classify']=='fp')).sum()
    fn_snp = ((~concordance['indel']) & (concordance['classify']=='fn')).sum()
    recall_snp = tp_snp/(tp_snp+fn_snp)
    precision_snp = tp_snp/(tp_snp+fp_snp)
    tp_indel = ((concordance['indel']) & (concordance['classify']=='tp')).sum()
    fp_indel = ((concordance['indel']) & (concordance['classify']=='fp')).sum()
    fn_indel = ((concordance['indel']) & (concordance['classify']=='fn')).sum()
    recall_indel = tp_indel/(tp_indel+fn_indel)
    precision_indel = tp_indel/(tp_indel+fp_indel)
    return pd.Series({"tp_snp":tp_snp,
                     "fp_snp":fp_snp,
                     "fn_snp":fn_snp,
                     "tp_indel":tp_indel,
                     "fp_indel": fp_indel, 
                     "fn_indel":fn_indel, 
                     "recall_snp":recall_snp, 
                     "precision_snp":precision_snp, 
                     "recall_indel":recall_indel, 
                     "precision_indel":precision_indel})