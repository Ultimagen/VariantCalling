from __future__ import annotations

import numpy as np
import pandas as pd
import pyfaidx
import pysam

import ugvc.filtering.multiallelics as mu
from ugvc.vcfbed import vcftools

SPAN_DEL = "*"


def split_multiallelic_variants_with_spandel(
    multiallelic_variant: pd.Series,
    spanning_deletion: pd.Series,
    call_vcf_header: pysam.VariantHeader | pysam.VariantFile | str,
    ref: pyfaidx.Fasta,
) -> pd.DataFrame:
    """Splits spanning deletion-containing multiallelic variants into multiple rows
    The process is similar to multiallelic variants, but the spanning deletion allele is always the second one

    Parameters
    ----------
    multiallelic_variant : pd.Series
        A row from the training set dataframe
    spanning_deletion: pd.Series
        The series that represents the spanning deletion
    call_vcf_header :
        Header of the VCF : pysam.VariantHeader | pysam.VariantFile | str
    ref : pyfaidx.Fasta
        Reference

    Returns
    -------
    pd.DataFrame
        A dataframe with the same columns as the training set dataframe
    """
    record_to_nbr_dict = vcftools.header_record_number(call_vcf_header)
    ignore_columns = ["gt_vcfeval", "alleles_vcfeval", "label", "sync"]

    for ic in ignore_columns:
        record_to_nbr_dict[ic] = 1

    alleles = multiallelic_variant["alleles"]
    spandel_idx = alleles.index(SPAN_DEL)
    # select the strongest alleles in order
    # **note that the spanning deletion allele is forced to be the less likely**
    allele_order = (
        np.argsort(
            [
                mu.select_pl_for_allele_subset(multiallelic_variant["pl"], (0, i), normed=False)[-1]
                + 100000 * (i == spandel_idx)
                for i in range(1, len(alleles))
            ]
        )
        + 1
    )
    return pd.concat(
        (
            extract_allele_subset_from_multiallelic_spanning_deletion(
                multiallelic_variant, spanning_deletion, alleles, record_to_nbr_dict, ref
            )
            for alleles in ((0, allele_order[0]), (allele_order[0], allele_order[1]))
        ),
        axis=1,
    ).T


def extract_allele_subset_from_multiallelic_spanning_deletion(
    multiallelic_variant: pd.Series,
    spanning_deletion: pd.Series,
    alleles: tuple,
    record_to_nbr_dict: dict,
    ref: pyfaidx.Fasta,
) -> pd.Series:
    """When analyzing multiallelic variants, we split them into pairs of alleles. Each pair of alleles
    need to have the updated columns (GT/PL etc.). This function updates those columns

    Parameters
    ----------
    multiallelic_variant : pd.Series
        The series that represent the multiallelic variant
    spanning_deletion: pd.Series
        The series that represent the spanning deletion variant (some values are extracted from it)
    alleles : tuple
        The allele tuple to fetch from the multiallelic variant
    record_to_nbr_dict : dict
        Dictionary that says what is the encoding in the VCF of each column
    ref : pyfaidx.Fasta
        Reference FASTA

    Returns
    -------
    pd.Series
        Updated subsetted variant
    """
    pos = multiallelic_variant["pos"]
    SPECIAL_TREATMENT_COLUMNS = {
        "sb": lambda x: x,
        "pl": lambda x: mu.select_pl_for_allele_subset(x, alleles),
        "gt": lambda x: mu.encode_gt_for_allele_subset(x, alleles),
        "ref": lambda _: mu.encode_ref_for_allele_subset(multiallelic_variant["alleles"], alleles),
        "indel": lambda _: mu.is_indel_subset(multiallelic_variant["alleles"], alleles, spandel=spanning_deletion),
        "x_ic": lambda _: mu.indel_classify_subset(multiallelic_variant["alleles"], alleles, spandel=spanning_deletion)[
            0
        ],
        "x_il": lambda _: mu.indel_classify_subset(multiallelic_variant["alleles"], alleles, spandel=spanning_deletion)[
            1
        ],
        "x_hil": lambda _: (
            mu.classify_hmer_indel_relative(
                multiallelic_variant["alleles"], alleles, ref, pos, flow_order="TGCA", spandel=spanning_deletion
            )[1],
        ),
        "x_hin": lambda _: (
            mu.classify_hmer_indel_relative(
                multiallelic_variant["alleles"], alleles, ref, pos, flow_order="TGCA", spandel=spanning_deletion
            )[0],
        ),
        "label": lambda x: mu.encode_label(x, alleles),
    }
    result = {}
    for col in multiallelic_variant.index:
        if col in SPECIAL_TREATMENT_COLUMNS:
            result[col] = SPECIAL_TREATMENT_COLUMNS[col](multiallelic_variant.at[col])
        elif isinstance(multiallelic_variant[col], tuple) and record_to_nbr_dict[col] != 1:
            result[col] = vcftools.subsample_to_alleles(multiallelic_variant.at[col], record_to_nbr_dict[col], alleles)
        else:
            result[col] = multiallelic_variant.at[col]
    return pd.Series(result)
