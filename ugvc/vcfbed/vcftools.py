# Functions to annotate/read operate on concordance dataframe
from __future__ import annotations

import os
from collections import defaultdict
from enum import Enum
import json

import numpy as np
import pandas as pd
import pysam
import subprocess
from tempfile import gettempdir

def get_vcf_df(
    variant_calls: str,
    sample_id: int = 0,
    sample_name: str = None,
    chromosome: str = None,
    scoring_field: str = None,
    ignore_fields: list = None,
) -> pd.DataFrame:
    """Reads VCF file into dataframe

    Parameters
    ----------
    variant_calls: str
        VCF file
    sample_id: int
        Index of sample to fetch (default: 0)
    sample_name: str
        Name of sample to fetch (default: None)
    chromosome: str
        Specific chromosome to load from vcf, (default: all chromosomes)
    scoring_field: str
        The name of the field that is used to score the variants.
        This value replaces the TREE_SCORE in the output data frame.
        When None TREE_SCORE is not replaced (default: None)
    ignore_fields: list, optional
        List of VCF tags to ignore (save memory)
    Returns
    -------
    pd.DataFrame
    """

    if ignore_fields is None:
        ignore_fields = []

    header = pysam.VariantFile(variant_calls).header
    if chromosome is None:
        variant_file = pysam.VariantFile(variant_calls)

    else:
        variant_file = pysam.VariantFile(variant_calls).fetch(chromosome)
    # pylint: disable-next=bad-builtin
    vfi = map(
        lambda x: defaultdict(
            lambda: None,
            x.info.items()
            + (
                x.samples[list(x.samples).index(sample_name)].items()
                if sample_name is not None
                else (x.samples[sample_id].items() if sample_id is not None else [])
            )
            + [
                ("QUAL", x.qual),
                ("CHROM", x.chrom),
                ("POS", x.pos),
                ("REF", x.ref),
                ("ID", x.id),
                ("ALLELES", x.alleles),
                ("FILTER", ";".join(x.filter.keys())),
            ],
        ),
        variant_file,
    )
    columns = [
        "GT",
        "PL",
        "DP",
        "AD",
        "MQ",
        "SOR",
        "AF",
        "DP_R",
        "DP_F",
        "AD_R",
        "AD_F",
        "TLOD",
        "VAF",
        "STRANDQ",
        "FPR",
        "GROUP",
        "TREE_SCORE",
        "VARIANT_TYPE",
        "DB",
        "AS_SOR",
        "AS_SORP",
        "FS",
        "VQR_VAL",
        "QD",
        "hiConfDeNovo",
        "loConfDeNovo",
        "GQ",
        "PGT",
        "PID",
        "PS",
        "AC",
        "AN",
        "BaseQRankSum",
        "ExcessHet",
        "MLEAC",
        "MLEAF",
        "MQRankSum",
        "ReadPosRankSum",
        "XC",
        "ID",
        "GNOMAD_AF",
        "NLOD",
        "NALOD",
        "X_IC",
        "X_IL",
        "X_HIL",
        "X_HIN",
        "X_LM",
        "X_RM",
        "X_GCC",
        "X_CSS",
        "RPA",
        "RU",
        "STR",
        "AVERAGE_TREE_SCORE",
        "VQSLOD",
        "BLACKLST",
        "SCORE"
    ]

    if scoring_field is not None and scoring_field not in columns:
        columns.append(scoring_field)

    columns = [x for x in columns if x in header.info.keys() + header.formats.keys()] + [
        "CHROM",
        "POS",
        "QUAL",
        "REF",
        "ALLELES",
        "FILTER",
        "ID",
    ]
    ignore_fields = [x.lower() for x in ignore_fields]
    columns = [x for x in columns if x.lower() not in ignore_fields]
    df = pd.DataFrame([[x[y] for y in columns] for x in vfi], columns=[x.lower() for x in columns])

    if scoring_field is not None:
        df["tree_score"] = df[scoring_field.lower()]

    df["indel"] = df["alleles"].apply(lambda x: len({len(y) for y in x}) > 1)

    df.index = [(x[1]["chrom"], x[1]["pos"]) for x in df.iterrows()]

    return df


def add_info_tag_from_df(
    vcf_input_file: str,
    vcf_output_file: str,
    df: pd.DataFrame,
    column: str,
    info_format: tuple,
):
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
            for row in vcfin:
                val = df.loc[[(row.chrom, row.start + 1)]][column].values[0]
                if val is None or val == "":
                    vcfout.write(row)
                else:
                    row.info[info_format[0]] = val
                    vcfout.write(row)


def get_region_around_variant(vpos: int, vlocs: np.ndarray, region_size: int = 100) -> tuple:
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
    if len(vlocs) == 0:
        return initial_region

    # expand the region to the left
    # clip for the cases when the variant is after all the variants and need
    # to be inserted at len(vlocs)
    while (
        vlocs[np.clip(np.searchsorted(vlocs, initial_region[0]), 0, len(vlocs)) - 1] - initial_region[0] < 10
        and vlocs[np.clip(np.searchsorted(vlocs, initial_region[0]), 0, len(vlocs)) - 1] - initial_region[0] >= 0
    ):
        initial_region = (initial_region[0] - 10, initial_region[1])

    initial_region = (max(initial_region[0], 0), initial_region[1])

    # expand the region to the right
    # The second conditions is for the case np.searchsorted(vlocs,
    # initial_region[1]) == 0
    while (
        initial_region[1] - vlocs[np.clip(np.searchsorted(vlocs, initial_region[1]), 1, len(vlocs)) - 1] < 10
        and initial_region[1] - vlocs[np.clip(np.searchsorted(vlocs, initial_region[1]), 1, len(vlocs)) - 1] >= 0
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

    center = (inspoints[1] - inspoints[0]) / 2
    distance = np.abs(variants.pos - center)
    take = np.argsort(distance)[:max_n_variants]
    return variants.iloc[take, :]


# Colors of the Bed
class FilteringColors(Enum):
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
        if "filter" in self.df.columns:
            row_cond = (self.df["classify"] == "fn") | ((self.df["classify"] == "tp") & (~self.filtering()))
            self.df = self.df.loc[row_cond, :]
        else:
            self.df = self.df.loc[(self.df["classify"] == "fn"), :]
        return self

    def get_fp(self):
        self.df = self.df.loc[self.df["classify"] == "fp", :]
        return self

    def get_tp(self):
        self.df = self.df.loc[self.df["classify"] == "tp", :]
        return self

    def get_fp_diff(self):
        row_cond = (self.df["classify"] == "tp") & (self.df["classify_gt"] == "fp")
        self.df = self.df.loc[row_cond, :]
        return self

    def get_fn_diff(self):
        row_cond = (self.df["classify"] == "tp") & (self.df["classify_gt"] == "fn")
        self.df = self.df.loc[row_cond, :]
        return self

    def get_snp(self):
        self.df = self.df.loc[~self.df["indel"], :]
        return self

    def get_h_mer(self, val_start: int = 1, val_end: int = 999):
        row_cond = (self.df["hmer_indel_length"] >= val_start) & (self.df["indel"])
        self.df = self.df.loc[row_cond, :]
        row_cond = self.df["hmer_indel_length"] <= val_end
        self.df = self.df.loc[row_cond, :]
        return self

    # we distinguish here two cases: insertion of a single
    # hmer (i.e. TG -> TCG), whcih is hmer indel but 0->1
    # and longer (i.e. TG -> TCAG) which is two errors and will be
    # called non-hmer indel
    def get_non_h_mer(self):
        row_cond = (self.df["hmer_indel_length"] == 0) & (self.df["indel"]) & (self.df["indel_length"] > 1)
        self.df = self.df.loc[row_cond, :]
        return self

    def get_h_mer_0(self):
        row_cond = (self.df["hmer_indel_length"] == 0) & (self.df["indel"]) & (self.df["indel_length"] == 1)
        self.df = self.df.loc[row_cond, :]
        return self

    def get_df(self):
        return self.df

    def filtering(self):
        do_filtering = "filter" in self.df.columns
        if not do_filtering:
            return pd.Series([True] * self.df.shape[0])
        filter_column = self.df["filter"]
        return ~filter_column.str.contains("LOW_SCORE", regex=False)

    def blacklist(self):
        do_filtering = "filter" in self.df.columns
        if not do_filtering:
            return pd.Series([False] * self.df.shape[0])
        filter_column = self.df["filter"]
        return filter_column.str.contains("BLACKLIST", regex=False)

    # for fp, we filter out all the low_score points, and color the lower 10% of them
    # in grey and the others in blue
    def filtering_fp(self) -> bool:

        if "tree_score" not in self.df.columns:
            self.df["tree_score"] = 1
        if not pd.isnull(self.df["tree_score"]).all() and pd.isnull(self.df["tree_score"]).any():
            self.df.loc[pd.isnull(self.df["tree_score"]), "tree_score"] = 1
        do_filtering = (
            "filter" in self.df.columns
            and "tree_score" in self.df.columns
            and (pd.to_numeric(self.df["tree_score"], errors="coerce").notnull().all())
        )
        if not do_filtering:
            return pd.Series([True] * self.df.shape[0])

        # in the new VCF format, the low_score is a separate column rather then a filter
        filter_column = self.df["filter"]
        # remove low score points
        self.df = self.df[filter_column.str.contains("PASS", regex=False)]

        tree_score_column = self.df["tree_score"]
        if len(tree_score_column) > 0:
            p_val = np.nanpercentile(tree_score_column, 10)
        else:
            p_val = 0
        # 10% of the points should be grey
        return tree_score_column > p_val

    # converts the h5 format to the BED format
    def bed_format(self, kind: str | None = None):
        do_filtering = "filter" in self.df.columns

        if do_filtering:
            if kind == "fp":
                rgb_color: pd.Series = self.filtering_fp()
            else:
                rgb_color = self.filtering()
            if kind == "fn":
                blacklist_color = self.blacklist()
            else:
                blacklist_color = np.zeros(self.df.shape[0], dtype=np.bool)
        hmer_length_column = self.df["hmer_indel_length"]
        # end pos
        # we want to add the rgb column, so we need to add all the columns
        # before it
        self.df = pd.concat(
            [
                self.df["chrom"],  # chrom
                self.df["pos"] - 1,  # chromStart
                self.df["pos"],  # chromEnd
                hmer_length_column,
            ],
            axis=1,
        )  # name

        self.df.columns = ["chrom", "chromStart", "chromEnd", "name"]

        # decide the color by filter column
        if do_filtering:
            fill_rgb = rgb_color.to_numpy()
            rgb_color[fill_rgb] = FilteringColors.CLEAR.value
            rgb_color[blacklist_color] = FilteringColors.BLACKLIST.value
            rgb_color[~fill_rgb] = FilteringColors.BORDERLINE.value
            rgb_color = list(rgb_color)
            self.df["score"] = 500
            self.df["strand"] = "."
            self.df["thickStart"] = self.df["chromStart"]
            self.df["thickEnd"] = self.df["chromEnd"]
            self.df["itemRgb"] = rgb_color
            self.df.columns = [
                "chrom",
                "chromStart",
                "chromEnd",
                "name",
                "score",
                "strand",
                "thickStart",
                "thickEnd",
                "itemRgb",
            ]
        self.df.sort_values(["chrom", "chromStart"], inplace=True)

        return self


def bed_files_output(data: pd.DataFrame, output_file: str, mode: str = "w", create_gt_diff: bool = True) -> None:
    """
    Create a set of bed file tracks that are often used in the
    debugging and the evaluation of the variant calling results

    Parameters
    ----------
    data : pd.DataFrame
        Concordance dataframe
    output_file: str
        Output file (extension will be split out and the result will serve as output prefix)
    mode: str
        Output file open mode ('w' or 'a')
    create_gt_diff: bool
        Create genotype diff files (False for Mutect, True for HC)
    """
    # pylint: disable-next=unused-variable
    basename, file_extension = os.path.splitext(output_file)

    # SNP filtering
    # fp
    snp_fp = FilterWrapper(data).get_snp().get_fp().bed_format(kind="fp").get_df()
    # fn
    snp_fn = FilterWrapper(data).get_snp().get_fn().bed_format(kind="fn").get_df()

    # Diff filtering
    if create_gt_diff:
        # fp
        all_fp_diff = FilterWrapper(data).get_fp_diff().bed_format(kind="fp").get_df()
        # fn
        all_fn_diff = FilterWrapper(data).get_fn_diff().bed_format(kind="fn").get_df()

    # Hmer filtering
    # 1 to 3
    # fp
    hmer_fp_1_3 = FilterWrapper(data).get_h_mer(val_start=1, val_end=3).get_fp().bed_format(kind="fp").get_df()
    # fn
    hmer_fn_1_3 = FilterWrapper(data).get_h_mer(val_start=1, val_end=3).get_fn().bed_format(kind="fn").get_df()

    # 4 until 7
    # fp
    hmer_fp_4_7 = FilterWrapper(data).get_h_mer(val_start=4, val_end=7).get_fp().bed_format(kind="fp").get_df()
    # fn
    hmer_fn_4_7 = FilterWrapper(data).get_h_mer(val_start=4, val_end=7).get_fn().bed_format(kind="fn").get_df()

    # 18 and more
    # fp
    hmer_fp_8_end = FilterWrapper(data).get_h_mer(val_start=8).get_fp().bed_format(kind="fp").get_df()
    # fn
    hmer_fn_8_end = FilterWrapper(data).get_h_mer(val_start=8).get_fn().bed_format(kind="fn").get_df()

    # non-Hmer filtering
    # fp
    non_hmer_fp = FilterWrapper(data).get_non_h_mer().get_fp().bed_format(kind="fp").get_df()
    # fn
    non_hmer_fn = FilterWrapper(data).get_non_h_mer().get_fn().bed_format(kind="fn").get_df()

    hmer_0_fp = FilterWrapper(data).get_h_mer_0().get_fp().bed_format(kind="fp").get_df()
    # fn
    hmer_0_fn = FilterWrapper(data).get_h_mer_0().get_fn().bed_format(kind="fn").get_df()

    def save_bed_file(file: pd.DataFrame, basename: str, curr_name: str, mode: str) -> None:
        if file.shape[0] > 0:
            file.to_csv(
                (basename + "_" + f"{curr_name}.bed"),
                sep="\t",
                index=False,
                header=False,
                mode=mode,
            )

    save_bed_file(snp_fp, basename, "snp_fp", mode)
    save_bed_file(snp_fn, basename, "snp_fn", mode)

    if create_gt_diff:
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
    out_pos = pos
    out_pos = out_pos > interval[0]
    out_pos = out_pos < interval[1]
    return out_pos


def genotype_ordering(num_alt: int) -> np.ndarray:
    # Returns a numpy array with the order of the genotypes (based on the section Genotype Ordering in the VCF spec).
    # Each row is a genotype, and the columns are the alleles.
    if num_alt == 1:
        gr_ar = np.array([[0, 0], [0, 1], [1, 1]])
    elif num_alt == 2:
        gr_ar = np.array([[0, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 2]])
    else:
        gr_ar = np.full([int((num_alt + 2) * (num_alt + 1) / 2), 2], fill_value=-1, dtype=int)
        i = 0
        for a2 in range(num_alt + 1):
            for a1 in range(a2 + 1):
                gr_ar[i, 0] = a1
                gr_ar[i, 1] = a2
                i += 1
    return gr_ar

def replace_data_in_specific_chromosomes(input_vcf: str, new_data_json: str, header_file: str, output_vcf:str, tempdir: str|None = None):
    """
    Takes a main vcf file, and a list of modified vcf files, broken down by chromosome (listed in the json).
    Replaces the data in the main vcf file by the data in the modified vcfs, and returns a concatenated vcf.
    
    Parameters
    ----------
    input_vcf: The main vcf which is being modified
    new_data_json: A json file in the form {chromosome: vcf}, with the data that will be replaced in input_json
    header_file: A file with a valid vcf header that will replace the current header. Make sure the new header fits
                 the new data.
    output_vcf: Name of the output vcf
    tempdir: Name of temp dir
    """
    if tempdir is None:
        tempdir = gettempdir()
    # Get the modified vcfs
    with open(new_data_json) as jfile:
        chrom2vcf = json.load(jfile)
    # Get info about the contigs
    contigs_info = {}
    with pysam.VariantFile(input_vcf) as in_vcf_obj:
        for k in in_vcf_obj.header.contigs.keys():
            contig = in_vcf_obj.header.contigs[k]
            contigs_info[contig.name] = {"id": contig.id, "length": contig.length}
    sorted_contigs = sorted(contigs_info.keys(), key=lambda k: contigs_info[k]["id"])
    # bed file with the chromosomes that were not modified by imputation
    untouched_contigs_bed = os.path.join(tempdir, "untouched_contigs.bed")
    with open(untouched_contigs_bed, "w", encoding="utf-8") as regions_file:
        sorted_contigs = sorted(contigs_info.keys(), key=lambda k: contigs_info[k]["id"])
        for contig in sorted_contigs:
            if contig not in chrom2vcf.keys():
                regions_file.write(contig + "\t1\t" + str(contigs_info[contig]["length"]) + "\n")
    # Extract chromosome without imputation data, and reheader to fit the header of vcfs after imputation
    untouched_contigs_vcf = os.path.join(tempdir, "untouched_contigs.vcf.gz")
    subprocess.check_call(f"bcftools view -O z -o {untouched_contigs_vcf} -R {untouched_contigs_bed} {input_vcf}", shell=True)
    untouched_contigs_rehead_vcf = os.path.join(tempdir, "untouched_contigs_rehead.vcf.gz")
    subprocess.check_call(f"bcftools reheader -h {header_file} -o {untouched_contigs_rehead_vcf} {untouched_contigs_vcf}", shell=True)
    # Merge
    vcfs_to_merge = list(chrom2vcf.values())
    vcfs_to_merge.append(untouched_contigs_rehead_vcf)
    subprocess.check_call(f"picard MergeVcfs INPUT={' INPUT='.join(vcfs_to_merge)} OUTPUT={output_vcf}", shell=True)
    subprocess.check_call(f"bcftools index -tf {output_vcf}", shell=True)
    