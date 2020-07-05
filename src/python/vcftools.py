import pysam
import pandas as pd
import pyfaidx
import tqdm
from python import utils
import numpy as np
from collections import defaultdict
import re

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

    concordance_df = pd.DataFrame(concordance)
    concordance_df.columns = ['chrom', 'pos', 'qual',
                              'ref', 'alleles', 'gt_calls', 'gt_ground_truth']

    concordance_df['indel'] = concordance_df['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    def classify(x):
        if x['gt_calls'] == (None, None):
            return 'fn'
        elif x['gt_ground_truth'] == (None, None):
            return 'fp'
        else:
            return 'tp'

    concordance_df['classify'] = concordance_df.apply(classify, axis=1)
    concordance_df.index = [(x[1]['chrom'], x[1]['pos'])
                            for x in concordance_df.iterrows()]
    vf = pysam.VariantFile(input_vcf)
    original = pd.DataFrame(
        [(x.chrom, x.pos, x.info['SOR']) for x in vf])
    original.columns = ['chrom', 'pos', 'sor']
    original.index = [(x[1]['chrom'], x[1]['pos'])
                      for x in original.iterrows()]
    concordance_df = concordance_df.join(original.drop(['chrom', 'pos'], axis=1))
    return concordance_df


def get_vcf_df(variant_calls: str, sample_id: int = 0) -> pd.DataFrame:
    '''Reads VCF file into dataframe, re

    Parameters
    ----------
    variant_calls: str
        VCF file
    sample_id: int
        Index of sample to fetch (default: 0)

    Returns
    -------
    pd.DataFrame
    '''
    vf = pysam.VariantFile(variant_calls)
    vfi = map(lambda x: defaultdict(lambda: None, x.info.items()
        + x.samples[sample_id].items() + [('QUAL', x.qual), ('CHROM', x.chrom), ('POS', x.pos), ('REF',x.ref),
          ('ALLELES', x.alleles), ('FILTER', ';'.join(x.filter.keys()))]), vf)

    columns = ['chrom', 'pos', 'qual',
               'ref', 'alleles', 'gt', 'pl',
               'dp', 'ad', 'mq', 'sor', 'af', 'filter',
               'dp_r','dp_f','ad_r','ad_f']
    concordance_df = pd.DataFrame([[x[y.upper()] for y in columns] for x in vfi])
    concordance_df.columns = columns

    concordance_df['indel'] = concordance_df['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    concordance_df.index = [(x[1]['chrom'], x[1]['pos'])
                            for x in concordance_df.iterrows()]
    return concordance_df

def add_info_tag_from_df( vcf_input_file: str, vcf_output_file: str, 
    df: pd.DataFrame, column: str, info_format: tuple): 
    """Adds a value of a dataframe column as an info field in the  VCF file 
    
    Parameters
    ----------
    vcf_input_file : str
        Input file
    vcf_output_file : str
        Output file
    df : pd.DataFrame
        Input dataframe
    column : str
        Column in he dataframe to fetch the data from
    info_format : tuple
        Tuple of info format (Tag, # values, format, Description)
    """
    with pysam.VariantFile(vcf_input_file) as vcfin: 
        hdr = vcfin.header
        if info_format[0] not in hdr.info.keys():
            hdr.info.add(*info_format)
        with pysam.VariantFile(vcf_output_file, mode="w", header=hdr) as vcfout : 
            for r in vcfin : 
                val = df.loc[[(r.chrom, r.start+1)]][column].values[0]
                if val is None or val == '':
                    vcfout.write(r)
                else: 
                    r.info[info_format[0]] = val
                    vcfout.write(r)

        
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


def fix_symbolic_alleles( df: pd.DataFrame) -> pd.DataFrame: 
    '''Replaces <%d> alleles with * for pysam compatibility

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    '''
    df['alleles'] = df['alleles'].apply(lambda x : tuple([y if re.match(r'<[0-9]+>',y) is None else '*' for y in x]))
    df['ref'] = df['ref'].apply(lambda x: x if re.match(r'<[0-9]+>',x) is None else '*')
    return df


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
    concordance['indel_length'] = concordance.apply(lambda x: abs(
        len(x['ref']) - max([len(y) for y in x['alleles'] if y != x['ref']])), axis=1)
    return concordance


def is_hmer_indel(concordance: pd.DataFrame, fasta_file: str) -> pd.DataFrame:
    '''Checks if the indel is hmer indel and outputs its length
    Note: The length of the shorter allele is output.

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
                return (utils.hmer_length(fasta_idx[rec['chrom']], rec['pos']), alt[0])

        if rec['indel_classify'] == 'del':
            del_seq = rec['ref'][1:]
            if len(set(del_seq)) != 1:
                return (0, None)
            elif fasta_idx[rec['chrom']][rec['pos'] + len(rec['ref']) - 1].seq.upper() != del_seq[0]:
                return (0, None)
            else:
                return (len(del_seq) + utils.hmer_length(fasta_idx[rec['chrom']],
                                                         rec['pos'] + len(rec['ref']) - 1), del_seq[0])
    results = concordance.apply(lambda x: _is_hmer(x, fasta_idx), axis=1)
    concordance['hmer_indel_length'] = [x[0] for x in results]
    concordance['hmer_indel_nuc'] = [x[1] for x in results]
    return concordance


def get_motif_around(concordance: pd.DataFrame, motif_size: int, fasta: str) -> pd.DataFrame:
    '''Extract sequence around the indel

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    motif_size: int
        Size of motif on each side
    fasta: str
        Indexed fasta

    Returns
    -------
    pd.DataFrame
        DataFrame. Adds "left_motif" and right_motif
    '''

    def _get_motif_around_snp(rec, size, faidx):
        chrom = faidx[rec['chrom']]
        pos = rec['pos']
        return chrom[pos - size - 1:pos - 1].seq.upper(), chrom[pos:pos + size].seq.upper()

    def _get_motif_around_non_hmer_indel(rec, size, faidx):
        chrom = faidx[rec['chrom']]
        pos = rec['pos']
        return chrom[pos - size:pos].seq.upper(),\
            chrom[pos + len(rec['ref']) - 1:pos + len(rec['ref']) - 1 + size].seq.upper()

    def _get_motif_around_hmer_indel(rec, size, faidx):
        chrom = faidx[rec['chrom']]
        pos = rec['pos']
        hmer_length = rec['hmer_indel_length']
        return chrom[pos - size:pos].seq.upper(), chrom[pos + hmer_length:pos + hmer_length + size].seq.upper()

    def _get_motif(rec, size, faidx):
        if rec['indel'] and rec['hmer_indel_length'] > 0:
            return _get_motif_around_hmer_indel(rec, size, faidx)
        elif rec['indel'] and rec['hmer_indel_length'] == 0:
            return _get_motif_around_non_hmer_indel(rec, size, faidx)
        else:
            return _get_motif_around_snp(rec, size, faidx)
    faidx = pyfaidx.Fasta(fasta)
    tmp = concordance.apply(lambda x: _get_motif(x, motif_size, faidx), axis=1)
    concordance['left_motif'] = [x[0] for x in list(tmp)]
    concordance['right_motif'] = [x[1] for x in list(tmp)]
    return concordance


def get_gc_content(concordance: pd.DataFrame, window_size: int, fasta: str) -> pd.DataFrame:
    '''Extract sequence around the indel

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    window_size: int
        Size of window for GC calculation (around start pos of variant)
    fasta: str
        Indexed fasta

    Returns
    -------
    pd.DataFrame
        DataFrame. Adds "left_motif" and right_motif
    '''
    def _get_gc(rec, size, faidx):
        chrom = faidx[rec['chrom']]
        beg = rec['pos'] - int(size / 2)
        end = beg + size
        seq = chrom[beg:end].seq.upper()
        seqGC = seq.replace('A', '').replace('T', '')
        return float(len(seqGC)) / len(seq)
    faidx = pyfaidx.Fasta(fasta)
    tmp = concordance.apply(lambda x: _get_gc(x, window_size, faidx), axis=1)
    concordance['gc_content'] = [x for x in list(tmp)]
    return concordance


def get_coverage(df: pd.DataFrame, alnfile: str, min_quality: int, filter_dups:bool=False):
    results = []
    with pysam.AlignmentFile(alnfile) as alns:
        for r in tqdm.tqdm(df.iterrows()):
            reads = alns.fetch(r[1].chrom, r[1].pos - 1, r[1].pos + 1)
            count_total = 0
            count_well = 0
            for read in reads:
                if filter_dups and read.is_duplicate: 
                        continue

                if read.mapping_quality > min_quality:
                    count_well += 1
                count_total += 1
            results.append((count_total, count_well))

    df['coverage'] = [x[0] for x in results]
    df['well_mapped_coverage'] = [x[1] for x in results]
    df['repetitive_read_coverage'] = [x[0] - x[1] for x in results]
    return df


def close_to_hmer_run(df: pd.DataFrame, runfile: str,
                      min_hmer_run_length: int=10, max_distance: int=10) -> pd.DataFrame:
    '''Adds column is_close_to_hmer_run and inside_hmer_run that is T/F'''
    df['close_to_hmer_run'] = False
    df['inside_hmer_run'] = False
    run_df = utils.parse_intervals_file(runfile, min_hmer_run_length)
    gdf = df.groupby('chrom')
    grun_df = run_df.groupby('chromosome')
    for chrom in gdf.groups.keys():
        gdf_ix = gdf.groups[chrom]
        grun_ix = grun_df.groups[chrom]
        pos1 = np.array(df.loc[gdf_ix, 'pos'])
        pos2 = np.array(run_df.loc[grun_ix, 'start'])
        pos1_closest_pos2_start = np.searchsorted(pos2, pos1) - 1
        close_dist = abs(pos1 - pos2[np.clip(pos1_closest_pos2_start, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(pos1_closest_pos2_start + 1, None, len(pos2) - 1)] - pos1) < max_distance
        pos2 = np.array(run_df.loc[grun_ix, 'end'])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)
        close_dist |= abs(pos1 - pos2[np.clip(pos1_closest_pos2_end - 1, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(pos1_closest_pos2_end, None, len(pos2) - 1)] - pos1) < max_distance
        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, "inside_hmer_run"] = is_inside
        df.loc[gdf_ix, "close_to_hmer_run"] = (close_dist & (~is_inside))
    return df


def annotate_intervals(df: pd.DataFrame, annotfile: str) -> pd.DataFrame:
    '''Adds column based on interval annotation file (T/F)'''
    annot = annotfile.split('/')[-1]
    if annot[-4:] == '.bed':
        annot = annot[:-4]
    print('Annotating ' + annot)

    df[annot] = False
    annot_df = utils.parse_intervals_file(annotfile)
    gdf = df.groupby('chrom')
    gannot_df = annot_df.groupby('chromosome')
    for chrom in gdf.groups.keys():
        gdf_ix = gdf.groups[chrom]
        gannot_ix = gannot_df.groups[chrom]
        pos1 = np.array(df.loc[gdf_ix, 'pos'])
        pos2 = np.array(annot_df.loc[gannot_ix, 'start'])
        pos1_closest_pos2_start = np.searchsorted(pos2, pos1) - 1
        pos2 = np.array(annot_df.loc[gannot_ix, 'end'])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)

        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, annot] = is_inside
    return df


def fill_filter_column(df: pd.DataFrame) -> pd.DataFrame:
    '''Fills filter status column with HPOL_RUN/PASS for false negatives

    Parameters
    ----------
    df : pd.DataFrame
        Description

    Returns
    -------
    pd.DataFrame
        Description
    '''
    if 'filter' not in df.columns:
        df['filter'] = np.nan
    fill_column_locs = pd.isnull(df['filter'])
    is_hpol = df.close_to_hmer_run | df.inside_hmer_run
    result = np.array(['HPOL_RUN'] * fill_column_locs.sum())
    result[~is_hpol[fill_column_locs]] = 'PASS'
    df.loc[fill_column_locs, 'filter'] = result
    return df

def annotate_cycle_skip(df: pd.DataFrame, flow_order: str, gt_field: str = None) -> pd.DataFrame : 
    """Adds cycle skip information: non-skip, NA, cycle-skip, possible cycle-skip

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    flow_order : str
        Flow order 
    gt_field: str
        If snps that correspond to a specific genotype are to be considered

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "cycleskip_status" - possible values are:
        * non-skip (not a cycle skip at any flow order)
        * NA (undefined - non-snps, multiallelic snps )
        * cycle-skip
        ( possible cycle-skip (cycle skip at a different flow order))
    """

    def is_multiallelic(x, gt_field): 
        s = set(x[gt_field]) | set([0])
        if len(s) > 2 :
            return True
        else : 
            return False

    def get_non_ref(x): 
        return [ y for y in x if y!=0][0]

    if gt_field is None : 
        na_pos = df['indel'] | (df['alleles'].apply(len) > 2) 
    else: 
        na_pos = df.apply(lambda x: is_multiallelic(x,gt_field), axis=1)

    df.loc[na_pos, 'cycleskip_status'] = "NA"
    snp_pos = ~na_pos
    snps = df.loc[snp_pos].copy()
    left_last = np.array(snps['left_motif']).astype(np.string_)
    right_first = np.array(snps['right_motif']).astype(np.string_)

    ref = np.array(snps['ref']).astype(np.string_)
    if gt_field is None  :
        alt = np.array(snps['alleles'].apply(lambda x:x[1] )).astype(np.string_)
    else : 
        nra = snps[gt_field].apply(get_non_ref)
        snps['nra_idx'] = nra
        snps.loc[snps.nra_idx.isnull(), 'nra_idx'] = 1
        snps['nra_idx'] = snps['nra_idx'].astype(np.int)
        alt = np.array(snps.apply(lambda x:x['alleles'][x['nra_idx']] , axis=1)).astype(np.string_)
        snps.drop('nra_idx', axis=1,inplace=True)

    ref_seqs = np.char.add(np.char.add(left_last, ref), right_first)
    alt_seqs = np.char.add(np.char.add(left_last, alt), right_first)

    ref_encs = [utils.generateKeyFromSequence(str(np.char.decode(x)), flow_order) for x in ref_seqs]
    alt_encs = [utils.generateKeyFromSequence(str(np.char.decode(x)), flow_order) for x in alt_seqs]

    cycleskip = np.array([x for x in range(len(ref_encs)) if len(ref_encs[x]) != len(alt_encs[x])])
    poss_cycleskip = [x for x in range(len(ref_encs)) if len(ref_encs[x]) == len(alt_encs[x])
                      and (np.any(ref_encs[x][ref_encs[x] - alt_encs[x] != 0] == 0) or
                           np.any(alt_encs[x][ref_encs[x] - alt_encs[x] != 0] == 0))]
    s = set(np.concatenate((cycleskip, poss_cycleskip)))
    non_cycleskip = [x for x in range(len(ref_encs)) if x not in s]

    vals = [''] * len(snps)
    for x in cycleskip:
        vals[x] = "cycle-skip"
    for x in poss_cycleskip:
        vals[x] = "possible-cycle-skip"
    for x in non_cycleskip:
        vals[x] = "non-skip"
    snps["cycleskip_status"] = vals

    df.loc[snp_pos, "cycleskip_status"] = snps["cycleskip_status"]
    return df
