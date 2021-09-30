import numpy as np
import pandas as pd
import pyfaidx
import python.utils as utils
import pysam
import tqdm
import pyBigWig as pbw
from typing import List
UNDETERMINED = "NA"


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
    concordance['indel_classify'] = concordance.apply(
        classify, axis=1, result_type='reduce')
    concordance['indel_length'] = concordance.apply(lambda x: max(
        [abs(len(y) - len(x['ref'])) for y in x['alleles']]), axis=1, result_type='reduce')
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
    results = concordance.apply(lambda x: _is_hmer(
        x, fasta_idx), axis=1, result_type='reduce')
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
            chrom[pos + len(rec['ref']) - 1:pos +
                  len(rec['ref']) - 1 + size].seq.upper()

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
    tmp = concordance.apply(lambda x: _get_motif(
        x, motif_size, faidx), axis=1, result_type='reduce')
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
    tmp = concordance.apply(lambda x: _get_gc(
        x, window_size, faidx), axis=1, result_type='reduce')
    concordance['gc_content'] = [x for x in list(tmp)]
    return concordance


def get_coverage(df: pd.DataFrame,
                 bw_coverage_files_high_quality: List[str],
                 bw_coverage_files_low_quality: List[str]) -> pd.DataFrame:
    """Adds coverage columns to the variant dataframe. Three columns are added: coverage - total coverage,
    well_mapped_coverage - coverage of reads with mapping quality > min_quality and repetitive_read_coverage -
    which is the difference between the two.
    bw_coverage_files should be outputs of `coverage_analysis.py` and have .chr??. inside the name

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (VCF or concordance)
    bw_coverage_files_high_quality : List[str]
        List of BW for the coverage at the high MAPQ quality threshold
    bw_coverage_files_low_quality : List[str]
        List of BW for the coverage at the MAPQ threshold 0


    Returns
    -------
    pd.DataFrame
        Modified dataframe
    """

    chrom2f = [(list(pbw.open(f).chroms().keys())[0], f)
               for f in bw_coverage_files_high_quality]
    chrom2bw_dict = dict(chrom2f)

    chrom2f = [(list(pbw.open(f).chroms().keys())[0], f)
               for f in bw_coverage_files_low_quality]
    for v in chrom2f:
        chrom2bw_dict[v[0]] = (chrom2bw_dict[v[0]], v[1])

    df.insert(len(df.columns), "coverage", np.NaN)
    df.insert(len(df.columns), "well_mapped_coverage", np.NaN)
    df.insert(len(df.columns), "repetitive_read_coverage", np.NaN)

    gdf = df.groupby("chrom")

    for g in gdf.groups:
        idx = gdf.groups[g]
        if g in chrom2bw_dict:
            bw_hq = chrom2bw_dict[g][0]
            bw_lq = chrom2bw_dict[g][1]
        else:
            continue

        with pbw.open(bw_hq) as bw:
            values_hq = [bw.values(g, x[1]-1, x[1])[0] for x in gdf.groups[g]]
        with pbw.open(bw_lq) as bw:
            values_lq = [bw.values(g, x[1]-1, x[1])[0] for x in gdf.groups[g]]
        df.loc[gdf.groups[g], "coverage"] = values_lq
        df.loc[gdf.groups[g], "well_mapped_coverage"] = values_hq
        df.loc[gdf.groups[g], "repetitive_read_coverage"] = np.array(
            values_lq) - np.array(values_hq)
    return df


def close_to_hmer_run(df: pd.DataFrame, runfile: str,
                      min_hmer_run_length: int = 10, max_distance: int = 10) -> pd.DataFrame:
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
        close_dist = abs(
            pos1 - pos2[np.clip(pos1_closest_pos2_start, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(
            pos1_closest_pos2_start + 1, None, len(pos2) - 1)] - pos1) < max_distance
        pos2 = np.array(run_df.loc[grun_ix, 'end'])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)
        close_dist |= abs(
            pos1 - pos2[np.clip(pos1_closest_pos2_end - 1, 0, None)]) < max_distance
        close_dist |= abs(
            pos2[np.clip(pos1_closest_pos2_end, None, len(pos2) - 1)] - pos1) < max_distance
        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, "inside_hmer_run"] = is_inside
        df.loc[gdf_ix, "close_to_hmer_run"] = (close_dist & (~is_inside))
    return df


def annotate_intervals(df: pd.DataFrame, annotfile: str) -> pd.DataFrame:
    '''
    Adds column based on interval annotation file (T/F)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be annotated
    annotfile: str
        bed file of the annotated intervals

    Returns
    -------
    pd.DataFrame
        Adds boolean column for the annotation (parsed from the file name)
    str
        Annotation name


    '''
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
    return df, annot


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
    # if there were run intervals
    if 'close_to_hmer_run' in df.columns:
        is_hpol = df.close_to_hmer_run | df.inside_hmer_run
    else:
        is_hpol = np.zeros(fill_column_locs.shape[0], dtype=np.bool)

    result = np.array(['HPOL_RUN'] * fill_column_locs.sum(), dtype=f'<U{len("HPOL_RUN")}')
    result[~is_hpol[fill_column_locs]] = 'PASS'
    df.loc[fill_column_locs, 'filter'] = result
    return df


def annotate_cycle_skip(df: pd.DataFrame, flow_order: str, gt_field: str = None) -> pd.DataFrame:
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
        if len(s) > 2:
            return True
        else:
            return False

    def get_non_ref(x):
        return [y for y in x if y != 0][0]

    def is_non_polymorphic(x, gt_field):
        s = set(x[gt_field]) | set([0])
        return len(s) == 1

    if gt_field is None:
        na_pos = df['indel'] | (df['alleles'].apply(len) > 2)
    else:
        na_pos = df['indel'] | df.apply(lambda x: is_multiallelic(
            x, gt_field), axis=1) | df.apply(lambda x: is_non_polymorphic(x, gt_field), axis=1)
    df['cycleskip_status'] = UNDETERMINED
    snp_pos = ~na_pos
    snps = df.loc[snp_pos].copy()
    left_last = np.array(snps['left_motif']).astype(np.string_)
    right_first = np.array(snps['right_motif']).astype(np.string_)

    ref = np.array(snps['ref']).astype(np.string_)
    if gt_field is None:
        alt = np.array(snps['alleles'].apply(
            lambda x: x[1] if len(x) > 1 else None)).astype(np.string_)
    else:
        nra = snps[gt_field].apply(get_non_ref)
        snps['nra_idx'] = nra
        snps.loc[snps.nra_idx.isnull(), 'nra_idx'] = 1
        snps['nra_idx'] = snps['nra_idx'].astype(np.int)
        alt = np.array(snps.apply(lambda x: x['alleles'][
                       x['nra_idx']], axis=1)).astype(np.string_)
        snps.drop('nra_idx', axis=1, inplace=True)

    ref_seqs = np.char.add(np.char.add(left_last, ref), right_first)
    alt_seqs = np.char.add(np.char.add(left_last, alt), right_first)

    ref_encs = [utils.catch(utils.generateKeyFromSequence,
                            str(np.char.decode(x)), flow_order, exception_type=ValueError,
                            handle=lambda x: UNDETERMINED) for x in ref_seqs]
    alt_encs = [utils.catch(utils.generateKeyFromSequence,
                            str(np.char.decode(x)), flow_order, exception_type=ValueError,
                            handle=lambda x: UNDETERMINED) for x in alt_seqs]

    cycleskip = np.array([x for x in range(len(ref_encs))
                          if type(ref_encs[x]) == np.ndarray and
                          type(alt_encs[x]) == np.ndarray and
                          len(ref_encs[x]) != len(alt_encs[x])])
    poss_cycleskip = [x for x in range(len(ref_encs)) if type(ref_encs[x]) == np.ndarray
                      and type(alt_encs[x]) == np.ndarray and
                      len(ref_encs[x]) == len(alt_encs[x])
                      and (np.any(ref_encs[x][ref_encs[x] - alt_encs[x] != 0] == 0) or
                           np.any(alt_encs[x][ref_encs[x] - alt_encs[x] != 0] == 0))]

    undetermined = np.array([x for x in range(
        len(ref_encs)) if (type(ref_encs[x]) == str and ref_encs[x] == UNDETERMINED) 
    or (type(alt_encs[x]) == str and alt_encs[x] == UNDETERMINED)])
    s = set(np.concatenate((cycleskip, poss_cycleskip, undetermined)))
    non_cycleskip = [x for x in range(len(ref_encs)) if x not in s]

    vals = [''] * len(snps)
    for x in cycleskip:
        vals[x] = "cycle-skip"
    for x in poss_cycleskip:
        vals[x] = "possible-cycle-skip"
    for x in non_cycleskip:
        vals[x] = "non-skip"
    for x in undetermined:
        vals[x] = UNDETERMINED
    snps["cycleskip_status"] = vals

    df.loc[snp_pos, "cycleskip_status"] = snps["cycleskip_status"]
    return df
