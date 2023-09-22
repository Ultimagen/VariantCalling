### This area of the code takes care of multiallelic variants and spanning deletions.
from __future__ import annotations

import numpy as np
import pandas as pd
import pyfaidx
import pysam

import ugvc.comparison.flow_based_concordance as fbc
import ugvc.flow_format.flow_based_read as fbr
import ugvc.vcfbed.vcftools as vcftools


def select_overlapping_variants(df: pd.DataFrame) -> list:
    """Selects lists of overlapping variants that need to be genotyped together. This
    can be multiallelic variants or variants with spanning deletion

    Parameters
    ----------
    df : pd.DataFrame
        Training set dataframe
    Returns
    -------
    list
        List of list of indices that generate the co-genotyped sets
    """

    multiallelic_locations = set(np.where(df["alleles"].apply(len) > 2)[0])
    del_length = df["alleles"].apply(lambda x: max(len(x[0]) - len(y) for y in x))
    current_span = 0
    results = []
    cluster = []
    for i in range(df.shape[0]):
        if len(cluster) == 0 and del_length[i] == 0:
            continue
        if df.iloc[i]["pos"] > current_span:
            if len(cluster) > 1:
                for c in cluster:
                    if c in multiallelic_locations:
                        multiallelic_locations.remove(c)
                results.append(cluster[:])
            cluster = []
        cluster.append(i)
        current_span = max(current_span, del_length[i] + df.iloc[i]["pos"])
    if len(cluster) > 1:
        for c in cluster:
            if c in multiallelic_locations:
                multiallelic_locations.remove(c)
        results.append(cluster[:])
    for m in multiallelic_locations:
        results.append([m])
    sorted_results = sorted(results)
    return sorted_results


def split_multiallelic_variants(
    multiallelic_variant: pd.Series, call_vcf_header: pysam.VariantHeader, ref: pyfaidx.Fasta
) -> pd.DataFrame:
    """Splits multiallelic variants into multiple rows

    Parameters
    ----------
    multiallelic_variant : pd.Series
        A row from the training set dataframe
    call_vcf_header: pysam.VariantHeader
        Header of the VCF
    ref: pyfaidx.Fasta
        Reference
    Returns
    -------
    pd.DataFrame
        A dataframe with the same columns as the training set dataframe
    """
    record_to_nbr_dict = vcftools.header_record_number(call_vcf_header)
    alleles = multiallelic_variant["alleles"]

    # select the strongest alleles in order
    allele_order = (
        np.argsort(
            [select_pl_for_allele_subset(multiallelic_variant["pl"], (0, i))[-1] for i in range(1, len(alleles))]
        )
        + 1
    )
    for alleles in [(0, allele_order[0]), (allele_order[0], allele_order[1])]:
        extract_allele_subset_from_multiallelic(multiallelic_variant, alleles, record_to_nbr_dict, ref)


def extract_allele_subset_from_multiallelic(
    multiallelic_variant: pd.Series, alleles: tuple, record_to_nbr_dict: dict, ref: pyfaidx.Fasta
) -> pd.Series:
    pos = multiallelic_variant["pos"]
    SPECIAL_TREATMENT_COLUMNS = {
        "pl": lambda x: select_pl_for_allele_subset(x, alleles),
        "gt": lambda x: encode_gt_for_allele_subset(x, alleles),
        "indel": lambda x: is_indel_subset(x, alleles),
        "x_ic": lambda x: indel_classify_subset(x, alleles),
        "x_hil": lambda x: classify_hmer_indel_relative(x, alleles, ref, pos, flow_order="TGCA")[0],
        "x_hin": lambda x: classify_hmer_indel_relative(x, alleles, ref, pos, flow_order="TGCA")[1],
    }
    result = {}
    for col in multiallelic_variant.columns:
        if col in SPECIAL_TREATMENT_COLUMNS:
            result[col] = SPECIAL_TREATMENT_COLUMNS[col](multiallelic_variant[col])
        elif type(multiallelic_variant[col]) == tuple:
            result[col] = vcftools.subsample_to_alleles(multiallelic_variant[col], record_to_nbr_dict[col], alleles)
        else:
            result[col] = multiallelic_variant[col]
    return pd.Series(result)


def encode_gt_for_allele_subset(original_gt: tuple, allele_idcs: tuple) -> tuple:
    """Encodes GT according to the subsampled allele indices.
    If only first allele is present in the original genotype: 0,0
    If only second allele is present in the original genotype: 1,1
    If both alleles are present: 0,1
    Requires: at least one allele is present

    Parameters
    ----------
    original_gt : tuple
        Original genotype tuple
    allele_idcs : tuple
        Indices of alleles that are being selected

    Returns
    -------
    tuple
        Resulting genotype, at least one allele should exist in the orginal genotype

    Raises
    ------
    RuntimeError
        if neither allele is present in the original genotype
    """
    assert (
        allele_idcs[0] in original_gt or allele_idcs[1] in original_gt
    ), "One of the alleles should be present in the GT"
    if allele_idcs[0] in original_gt and allele_idcs[1] not in original_gt:
        return (0, 0)
    if allele_idcs[1] in original_gt and allele_idcs[0] not in original_gt:
        return (1, 1)
    if allele_idcs[1] in original_gt and allele_idcs[0] in original_gt:
        return (0, 1)
    raise RuntimeError("Neither allele found in the original genotype")


def _get_pl_idx(tup: tuple) -> int:
    """Returns the index of the PL value in the tuple

    Parameters
    ----------
    tup : tuple
        PL tuple

    Returns
    -------
    int
        Index of the PL value
    """
    offset: int = max(tup) * (max(tup) + 1) // 2
    return offset + int(min(tup))


def select_pl_for_allele_subset(original_pl: tuple, allele_idcs: tuple) -> tuple:
    """Selects Pl values according to the subsampled allele indices. Normalizes to the minimum

    Parameters
    ----------
    original_pl : tuple
        Original PL tuple
    allele_idcs : tuple
        Indices of alleles that are being selected

    Returns
    -------
    tuple
        Resulting genotype, at least one allele should exist in the orginal genotype
    """
    allele_idcs = tuple(sorted(allele_idcs))
    idcs = [
        _get_pl_idx(x)
        for x in ((allele_idcs[0], allele_idcs[0]), (allele_idcs[0], allele_idcs[1]), (allele_idcs[1], allele_idcs[1]))
    ]
    pltake = [original_pl[x] for x in idcs]
    min_pl = min(pltake)
    return tuple(x - min_pl for x in pltake)


def is_indel_subset(alleles: tuple, allele_indices: tuple) -> bool:
    """Checks if the variant is an indel

    Parameters
    ----------
    alleles : tuple
        Alleles tuple
    allele_indices : tuple
        Indices of alleles that are being selected

    Returns
    -------
    bool
        True if indel, False otherwise
    """
    return any(len(alleles[x]) != len(alleles[allele_indices[0]]) for x in allele_indices)


def indel_classify_subset(alleles: tuple, allele_indices: tuple) -> tuple[tuple[str | None], tuple[int | None]]:
    """Checks if the variant is insertion or deletion

    Parameters
    ----------
    alleles : tuple
        Alleles tuple
    allele_indices : tuple
        Indices of alleles that are being selected

    Returns
    -------
    tuple[str|None]
        ('ins',) or ('del',)
    """
    if not is_indel_subset(alleles, allele_indices):
        return ((None,), (None,))
    ref_allele = alleles[allele_indices[0]]
    alt_allele = alleles[allele_indices[1]]
    if len(ref_allele) > len(alt_allele):
        return (("del",), (len(ref_allele) - len(alt_allele),))
    return (("ins",), (len(alt_allele) - len(ref_allele),))


def classify_hmer_indel_relative(
    alleles: tuple, allele_indices: tuple, ref: pyfaidx.Fasta, pos: int, flow_order: str = "TGCA"
) -> tuple:
    """Checks if one allele is hmer indel relative to the other

    Parameters
    ----------
    alleles : tuple
        Tuple of all alleles
    allele_indices : tuple
        Pair of allele indices, the second one is measured relative to the first one
    ref : pyfaidx.Fasta
        Reference sequence
    pos: int
        Position of the variant (one-based)
    flow_order: str
        Flow order, default is TGCA
    Returns
    -------
    tuple
        Pair of hmer indel length and hmer indel nucleotide
    """
    refstr = fbc.get_reference_from_region(ref, (pos - 10, pos + 10))
    refstr = refstr.upper()
    fake_series = pd.DataFrame(pd.Series({"alleles": alleles, "gt": allele_indices, "pos": pos, "ref": alleles[0]})).T
    haplotypes = fbc.apply_variants_to_reference(refstr, fake_series, pos - 10, genotype_col="gt", include_ref=False)
    fhaplotypes = [fbr.generate_key_from_sequence(x, flow_order) for x in haplotypes]
    compare = fbc.compare_haplotypes(fhaplotypes[0:1], fhaplotypes[1:2])
    if compare[0] != 1:
        return (0, None)
    flow_location = np.nonzero(fhaplotypes[0] - fhaplotypes[1])[0]
    nucleotide = flow_order[flow_location[0] % 4]
    length = min(int(fhaplotypes[0][flow_location]), int(fhaplotypes[1][flow_location]))
    return (nucleotide, length)


# special treatment
# rpa
