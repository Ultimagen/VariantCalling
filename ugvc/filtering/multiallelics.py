from __future__ import annotations

import numpy as np
import pandas as pd
import pyfaidx
import pysam

import ugvc.comparison.flow_based_concordance as fbc
import ugvc.filtering.training_prep as tprep
import ugvc.flow_format.flow_based_read as fbr
from ugvc.vcfbed import vcftools

SPAN_DEL = "*"


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
        # Fixing notorious cases that happen in the GATK when the variant overlaps deletion but does not contain it.
        # not sure why this is happening
        first = True
        non_span = []
        for c in cluster:
            if not first and SPAN_DEL not in df.iloc[c]["alleles"]:
                non_span.append(c)
            first = False
        cluster = [x for x in cluster if x not in non_span]
        if len(cluster) > 1:
            for c in cluster:
                if c in multiallelic_locations:
                    multiallelic_locations.remove(c)
            results.append(cluster)
    for m in multiallelic_locations:
        results.append([m])
    sorted_results = sorted(results)
    return sorted_results


def split_multiallelic_variants(
    multiallelic_variant: pd.Series, call_vcf_header: pysam.VariantHeader | pysam.VariantFile | str, ref: pyfaidx.Fasta
) -> pd.DataFrame:
    """Splits multiallelic variants into multiple rows

    Parameters
    ----------
    multiallelic_variant : pd.Series
        A row from the training set dataframe
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

    # select the strongest alleles in order
    allele_order = (
        np.argsort(
            [
                select_pl_for_allele_subset(multiallelic_variant["pl"], (0, i), normed=False)[-1]
                for i in range(1, len(alleles))
            ]
        )
        + 1
    )
    return pd.concat(
        (
            extract_allele_subset_from_multiallelic(multiallelic_variant, alleles, record_to_nbr_dict, ref)
            for alleles in ((0, allele_order[0]), (allele_order[0], allele_order[1]))
        ),
        axis=1,
    ).T


def extract_allele_subset_from_multiallelic(
    multiallelic_variant: pd.Series, alleles: tuple, record_to_nbr_dict: dict, ref: pyfaidx.Fasta
) -> pd.Series:
    pos = multiallelic_variant["pos"]
    SPECIAL_TREATMENT_COLUMNS = {
        "sb": lambda x: x,
        "pl": lambda x: select_pl_for_allele_subset(x, alleles),
        "gt": lambda x: encode_gt_for_allele_subset(x, alleles),
        "ref": lambda _: encode_ref_for_allele_subset(multiallelic_variant["alleles"], alleles),
        "indel": lambda _: is_indel_subset(multiallelic_variant["alleles"], alleles),
        "x_ic": lambda _: indel_classify_subset(multiallelic_variant["alleles"], alleles)[0],
        "x_il": lambda _: indel_classify_subset(multiallelic_variant["alleles"], alleles)[1],
        "x_hil": lambda _: (
            classify_hmer_indel_relative(multiallelic_variant["alleles"], alleles, ref, pos, flow_order="TGCA")[1],
        ),
        "x_hin": lambda _: (
            classify_hmer_indel_relative(multiallelic_variant["alleles"], alleles, ref, pos, flow_order="TGCA")[0],
        ),
        "label": lambda x: encode_label(x, alleles),
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


def encode_ref_for_allele_subset(allele_tuple: tuple[str], allele_idcs: tuple) -> str:
    """Make reference allele for the indices tuple

    Parameters
    ----------
    allele_tuple: tuple
        Tuple of alleles
    allele_idcs:
        Indices to select (the first one will be considered "ref")

    Returns
    -------
    str
        Indices to select (the first one will be considered "ref")

    Returns
    -------
    str
        String that corresponds to the reference allele
    """
    return allele_tuple[allele_idcs[0]]


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


def select_pl_for_allele_subset(original_pl: tuple, allele_idcs: tuple, normed: bool = True) -> tuple:
    """Selects Pl values according to the subsampled allele indices. Normalizes to the minimum

    Parameters
    ----------
    original_pl : tuple
        Original PL tuple
    allele_idcs : tuple
        Indices of alleles that are being selected
    normed: bool
        Should PL be normalized after subsetting so that the highest is zero, default: True

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
    if normed:
        min_pl = min(pltake)
        return tuple(x - min_pl for x in pltake)
    return tuple(pltake)


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


def encode_label(original_label: tuple, allele_indices: tuple) -> tuple:
    """Encodes a training label for the subset of alleles
    Parameters
    ----------
    original_label: tuple
        The current label
    allele_indices: tuple
        Pair of allele indices, the second one is measured relative to the first

    Returns
    -------
    tuple:
        label - (0,0),(0,1),(1,1). In case the label can't be encoded in the current alleles - for example when
        both alleles are not in the label - returns (-2,-2)

    Raises
    ------
    RuntimeError
        If there is some issue encoding the label
    """

    # TODO: check what MISS is coming from (maybe these should be false positives)?
    if tprep.MISS in original_label or tprep.IGNORE in original_label:
        return original_label
    if allele_indices[0] in original_label and allele_indices[1] in original_label:
        return (0, 1)
    if allele_indices[0] in original_label and allele_indices[1] not in original_label:
        return (0, 0)
    if allele_indices[1] in original_label and allele_indices[0] not in original_label:
        return (1, 1)
    if allele_indices[0] not in original_label and allele_indices[1] not in original_label:
        return (tprep.MISS, tprep.MISS)
    raise RuntimeError(f"can''t encode {original_label} with {allele_indices} Bug?")


def cleanup_multiallelics(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes multiallelics in the training set dataframe. Converts non-h-indels that
        are hmer indels into h-indel variant type. Adjust the values of the RU/RPA/STR
    when the variant is convered to hmer indel. (I.e. A -> CCA or CA, CCA is hmer indel of CA).

        Parameters
        ----------
        df : pd.DataFrame
           Input data frame

        Returns
        -------
        pd.DataFrame
            Output dataframe
    """
    df = df.copy()
    select = (df["variant_type"] == "non-h-indel") & (df["x_hil"].apply(lambda x: x[0] is not None))
    df.loc[select, "variant_type"] = "h-indel"
    df.loc[select, "str"] = True
    df.loc[select, "ru"] = df.loc[select, "x_hin"].apply(lambda x: x[0])
    ins_or_del = df.loc[select, "x_ic"].apply(lambda x: x[0])
    df.loc[select, "ins_or_del"] = ins_or_del

    def _alleles_lengths(v: pd.Series) -> tuple:
        if v["ins_or_del"] == "ins":
            return (v["x_hil"][0], v["x_hil"][0] + v["x_il"][0])
        return (v["x_hil"][0] + v["x_il"][0], v["x_hil"][0])

    df.loc[select, "rpa"] = df.loc[select].apply(_alleles_lengths, axis=1)
    df.drop("ins_or_del", axis=1, inplace=True)
    select = (df["variant_type"] == "h-indel") & (df["x_hil"].apply(lambda x: x[0] is None))
    df.loc[select, "variant_type"] = "non-h-indel"

    return df
