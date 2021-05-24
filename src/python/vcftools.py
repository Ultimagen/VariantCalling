# Functions to annotate/read operate on concordance dataframe
import pysam
import pandas as pd
import os
import numpy as np
from collections import defaultdict
from enum import Enum


def get_vcf_df(variant_calls: str, sample_id: int = 0, chromosome: str = None) -> pd.DataFrame:
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

    if chromosome is None:
        vf = pysam.VariantFile(variant_calls)
    else:
        vf = pysam.VariantFile(variant_calls).fetch(chromosome)

    vfi = map(lambda x: defaultdict(lambda: None, x.info.items()
                                    + x.samples[sample_id].items() +
                                    [('QUAL', x.qual), ('CHROM', x.chrom), ('POS', x.pos), ('REF', x.ref), ('ID', x.id),
                                     ('ALLELES', x.alleles), ('FILTER', ';'.join(x.filter.keys()))]), vf)

    # columns = ['chrom', 'pos', 'qual',
    #            'ref', 'alleles', 'gt', 'pl',
    #            'dp', 'ad', 'mq', 'sor', 'af', 'filter',
    #            'dp_r', 'dp_f', 'ad_r', 'ad_f', 'tlod', 'strandq','fpr','tree_score','variant_type','db']
    # concordance_df = pd.DataFrame([[x[y.upper()] for y in columns] for x in vfi])
    # concordance_df.columns = columns

    columns = ['CHROM', 'POS', 'QUAL',
               'REF', 'ALLELES', 'GT', 'PL',
               'DP', 'AD', 'MQ', 'SOR', 'AF', 'FILTER',
               'DP_R', 'DP_F', 'AD_R', 'AD_F', 'TLOD', 'STRANDQ','FPR', 'GROUP','TREE_SCORE','VARIANT_TYPE','DB',
               'AS_SOR', 'AS_SORP', 'FS', 'VQR_VAL', 'QD',
               'GQ', 'PGT', 'PID', 'PS',
               'AC', 'AN', 'BaseQRankSum','ExcessHet', 'MLEAC', 'MLEAF', 'MQRankSum', 'ReadPosRankSum','XC','ID','GNOMAD_AF']
    concordance_df = pd.DataFrame([[x[y] for y in columns] for x in vfi],columns=[x.lower() for x in columns])


    concordance_df['indel'] = concordance_df['alleles'].apply(
        lambda x: len(set(([len(y) for y in x]))) > 1)

    concordance_df.index = [(x[1]['chrom'], x[1]['pos'])
                            for x in concordance_df.iterrows()]


    return concordance_df


def add_info_tag_from_df(vcf_input_file: str, vcf_output_file: str,
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
        with pysam.VariantFile(vcf_output_file, mode="w", header=hdr) as vcfout:
            for r in vcfin:
                val = df.loc[[(r.chrom, r.start + 1)]][column].values[0]
                if val is None or val == '':
                    vcfout.write(r)
                else:
                    r.info[info_format[0]] = val
                    vcfout.write(r)


def get_region_around_variant(
    vpos: int, vlocs: np.ndarray, region_size: int = 100
) -> tuple:
    """Finds a genomic region around `vpos` of length at
    least `region_size` bases. `vlocs` is a list of locations.
    There is no location from vlocs that falls inside the region
    that is closer than 10 bases to either end of the region.
    This function is useful for finding region around a variant.

    Parameters
    ----------
    vpos : int
        Position of the center of the region
    vlocs : np.ndarray
        Variant locations that need to be distant from the ends
    region_size : int, optional
        Initial size of the region around `vpos`

    Returns
    ------------------
    tuple
        (start, end)
    """
    initial_region = (max(vpos - region_size // 2, 0), vpos + region_size // 2)

    # expand the region to the left
    # clip for the cases when the variant is after all the variants and need
    # to be inserted at len(vlocs)
    while (
        vlocs[np.clip(np.searchsorted(
            vlocs, initial_region[0]), 0, len(vlocs)) - 1] - initial_region[0] < 10
        and
        vlocs[np.clip(np.searchsorted(vlocs, initial_region[0]), 0, len(vlocs)) - 1] - initial_region[0] >= 0
    ):
        initial_region = (initial_region[0] - 10, initial_region[1])

    initial_region = (max(initial_region[0], 0), initial_region[1])

    # expand the region to the right
    # The second conditions is for the case np.searchsorted(vlocs,
    # initial_region[1]) == 0
    while (
        initial_region[1] -
            vlocs[np.clip(np.searchsorted(
                vlocs, initial_region[1]), 1, len(vlocs)) - 1] < 10
            and initial_region[1] -
            vlocs[np.clip(np.searchsorted(
                vlocs, initial_region[1]), 1, len(vlocs)) - 1]
            >= 0
    ):
        initial_region = (initial_region[0], initial_region[1] + 10)

    return initial_region


def get_variants_from_region(variant_df: pd.DataFrame, region: tuple, max_n_variants: int = 10) -> pd.DataFrame:
    """Returns variants from `variant_df` contained in the `region`

    Parameters
    ----------
    variant_df : pd.DataFrame
        Dataframe of the variants
    region : tuple
        (start, end)
    max_n_variants: int
        Maximal number of variants to extract from the region (default: 10), 
        variants further from the center will be removed first

    Returns
    -------
    pd.DataFrame
        subframe of variants contained in the region
    """
    inspoints = np.searchsorted(variant_df.pos, region)
    variants = variant_df.iloc[np.arange(*inspoints), :]
    if variants.shape[0] <= max_n_variants:
        return variants
    else:
        center = (inspoints[1] - inspoints[0]) / 2
        distance = np.abs(variants.pos - center)
        take = np.argsort(distance)[:max_n_variants]
        return variants.iloc[take, :]


# Colors of the Bed
class FilteringColors (Enum):
    BLACKLIST = "0,255,0"
    BORDERLINE = "121,121,121"
    CLEAR = "255,0,0"


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
                (self.df['classify'] == 'fn') | ((self.df['classify'] == 'tp') & (~ self.filtering()))]
        else:
            self.df = self.df[(self.df['classify'] == 'fn')]
        return self

    def get_fp(self):
        self.df = self.df[self.df['classify'] == 'fp']
        return self

    def get_tp(self):
        self.df = self.df[self.df['classify'] == 'tp']
        return self

    def get_fp_diff(self):
        self.df = self.df[(self.df['classify'] == 'tp') &
                          (self.df['classify_gt'] == 'fp')]
        return self

    def get_fn_diff(self):
        self.df = self.df[((self.df['classify'] == 'tp') &
                           (self.df['classify_gt'] == 'fn'))]
        return self

    def get_SNP(self):
        self.df = self.df[self.df['indel'] == False]
        return self

    def get_h_mer(self, val_start: int = 1, val_end: int = 999):
        self.df = self.df[(self.df['hmer_indel_length'] >=
                           val_start) & (self.df['indel'] == True)]
        self.df = self.df[(self.df['hmer_indel_length'] <= val_end)]
        return self

    # we distinguish here two cases: insertion of a single 
    # hmer (i.e. TG -> TCG), whcih is hmer indel but 0->1
    # and longer (i.e. TG -> TCAG) which is two errors and will be 
    # called non-hmer indel
    def get_non_h_mer(self):
        self.df = self.df[(self.df['hmer_indel_length'] == 0)
                          & (self.df['indel'] == True) &
                          (self.df['indel_length']>1)]
        return self

    def get_h_mer_0(self):
        self.df = self.df[(self.df['hmer_indel_length'] == 0)
                          & (self.df['indel'] == True) &
                          (self.df['indel_length']==1)]
        return self

    def get_df(self):
        return self.df

    def filtering(self):
        do_filtering = 'filter' in self.df.columns
        if not do_filtering:
            return pd.Series([True] * self.df.shape[0])
        filter_column = self.df['filter']
        return ~filter_column.str.contains('LOW_SCORE', regex=False)

    def blacklist(self):
        do_filtering = 'filter' in self.df.columns
        if not do_filtering:
            return pd.Series([False] * self.df.shape[0])
        filter_column = self.df['filter']
        return filter_column.str.contains('BLACKLIST', regex=False)

    # for fp, we filter out all the low_score points, and color the lower 10% of them
    # in grey and the others in blue
    def filtering_fp(self):
        do_filtering = 'filter' in self.df.columns \
                       and 'tree_score' in self.df.columns \
                       and (pd.to_numeric(self.df['tree_score'], errors='coerce').notnull().all())
        if not do_filtering:
            return pd.Series([True] * self.df.shape[0])

        # in the new VCF format, the low_score is a separate column rather then a filter 
        filter_column = self.df['filter']
        # remove low score points
        self.df = self.df[~filter_column.str.contains(
            'LOW_SCORE', regex=False)]

        tree_score_column = self.df['tree_score']
        if len(tree_score_column) > 0:
            p = np.nanpercentile(tree_score_column, 10)
        else:
            p = 0
        # 10% of the points should be grey
        return tree_score_column > p

    # converts the h5 format to the BED format
    def BED_format(self, kind=None):
        do_filtering = 'filter' in self.df.columns
        if do_filtering:
            if kind == "fp":
                rgb_color = self.filtering_fp()
            else:
                rgb_color = self.filtering()
            if kind == "fn":
                blacklist_color = self.blacklist()
            else:
                blacklist_color = np.zeros(self.df.shape[0], dtype=np.bool)

        hmer_length_column = self.df['hmer_indel_length']
        # end pos
        # we want to add the rgb column, so we need to add all the columns
        # before it
        self.df = pd.concat([self.df['chrom'],  # chrom
                             self.df['pos'] - 1,  # chromStart
                             self.df['pos'],  # chromEnd
                             hmer_length_column], axis=1)  # name

        self.df.columns = ['chrom', 'chromStart', 'chromEnd', 'name']

        # decide the color by filter column
        if do_filtering:
            rgb_color[rgb_color] = FilteringColors.CLEAR.value
            rgb_color[blacklist_color] = FilteringColors.BLACKLIST.value
            rgb_color[rgb_color == False] = FilteringColors.BORDERLINE.value
            rgb_color = list(rgb_color)
            self.df['score'] = 500
            self.df['strand'] = "."
            self.df['thickStart'] = self.df['chromStart']
            self.df['thickEnd'] = self.df['chromEnd']
            self.df['itemRgb'] = rgb_color
            self.df.columns = ['chrom', 'chromStart', 'chromEnd', 'name',
                               'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
        self.df.sort_values(['chrom', 'chromStart'], inplace=True)
        return self


def bed_files_output(data: pd.DataFrame, output_file: str, mode: str = 'w', create_gt_diff: bool = True) -> None:
    '''Create a set of bed file tracks that are often used in the
    debugging and the evaluation of the variant calling results

    Parameters
    ----------
    df : pd.DataFrame
        Concordance dataframe
    output_file: str
        Output file (extension will be split out and the result will serve as output prefix)
    mode: str
        Output file open mode ('w' or 'a')
    create_gt_diff: bool
        Create genotype diff files (False for Mutect, True for HC)

    Returns
    -------
    None
    '''

    basename, file_extension = os.path.splitext(output_file)

    # SNP filtering
    # fp
    snp_fp = FilterWrapper(data).get_SNP(
    ).get_fp().BED_format(kind="fp").get_df()
    # fn
    snp_fn = FilterWrapper(data).get_SNP().get_fn().BED_format(kind="fn").get_df()

    # Diff filtering
    if create_gt_diff:
        # fp
        all_fp_diff = FilterWrapper(data).get_fp_diff().BED_format(kind="fp").get_df()
        # fn
        all_fn_diff = FilterWrapper(data).get_fn_diff().BED_format(kind="fn").get_df()

    # Hmer filtering
    # 1 to 3
    # fp
    hmer_fp_1_3 = FilterWrapper(data).get_h_mer(
        val_start=1, val_end=3).get_fp().BED_format(kind="fp").get_df()
    # fn
    hmer_fn_1_3 = FilterWrapper(data).get_h_mer(
        val_start=1, val_end=3).get_fn().BED_format(kind="fn").get_df()

    # 4 until 7
    # fp
    hmer_fp_4_7 = FilterWrapper(data).get_h_mer(
        val_start=4, val_end=7).get_fp().BED_format(kind="fp").get_df()
    # fn
    hmer_fn_4_7 = FilterWrapper(data).get_h_mer(val_start=4, val_end=7).get_fn().BED_format(kind="fn"
    ).get_df()

    # 18 and more
    # fp
    hmer_fp_8_end = FilterWrapper(data).get_h_mer(
        val_start=8).get_fp().BED_format(kind="fp").get_df()
    # fn
    hmer_fn_8_end = FilterWrapper(data).get_h_mer(
        val_start=8).get_fn().BED_format(kind="fn").get_df()

    # non-Hmer filtering
    # fp
    non_hmer_fp = FilterWrapper(data).get_non_h_mer(
    ).get_fp().BED_format(kind="fp").get_df()
    # fn
    non_hmer_fn = FilterWrapper(
        data).get_non_h_mer().get_fn().BED_format(kind="fn").get_df()

    hmer_0_fp = FilterWrapper(data).get_h_mer_0(
    ).get_fp().BED_format(kind="fp").get_df()
    # fn
    hmer_0_fn = FilterWrapper(
        data).get_h_mer_0().get_fn().BED_format(kind="fn").get_df()

    def save_bed_file(file: pd.DataFrame, basename: str, curr_name: str, mode: str) -> None:
        if file.shape[0]>0:
            file.to_csv((basename + "_" + f"{curr_name}.bed"), sep='\t', index=False, header=False, mode=mode)

    save_bed_file(snp_fp, basename, "snp_fp", mode)
    save_bed_file(snp_fn, basename, "snp_fn", mode)

    if create_gt_diff : 
        save_bed_file(all_fp_diff, basename, "genotyping_errors_fp", mode=mode)
        save_bed_file(all_fn_diff, basename, "genotyping_errors_fn", mode=mode)

    save_bed_file(hmer_fp_1_3, basename, "hmer_fp_1_3", mode=mode)
    save_bed_file(hmer_fn_1_3, basename, "hmer_fn_1_3", mode=mode)
    save_bed_file(hmer_fp_4_7, basename, "hmer_fp_4_7", mode=mode)
    save_bed_file(hmer_fn_4_7, basename, "hmer_fn_4_7", mode=mode)
    save_bed_file(hmer_fp_8_end, basename, "hmer_fp_8_end", mode=mode)
    save_bed_file(hmer_fn_8_end, basename, "hmer_fn_8_end", mode=mode)

    save_bed_file(non_hmer_fp, basename, "non_hmer_fp", mode=mode)
    save_bed_file(non_hmer_fn, basename, "non_hmer_fn", mode=mode)

    save_bed_file(hmer_0_fp, basename, "hmer_0_fp", mode)
    save_bed_file(hmer_0_fn, basename, "hmer_0_fn", mode)


def isin(pos, interval):
    return pos > interval[0] and pos < interval[1]
