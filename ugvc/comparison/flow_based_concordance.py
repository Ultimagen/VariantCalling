"""Module for flow-based reinterpretation of variants

Genotype comparison between the Ultima data and the ground truth
from Illumina has a specific challenge: hmer indels connected to
the real variants generate sometimes variant at a different position
For instance, ground truth
`CATATG -> CATATATG`
is interpreted as `C -> CAT` in the VCF
However, the combination of the insertion AT with false positive
hmer indel `G->GG` is read in the VCF as
`T -> TATG`

This generates variants on different positions between the ground truth
and the VCF and is interpreted as false positive non-hmer indel + false
negative non-hmer indel.

This module provides functionality to modify concordance errors the most
likely error that we made **relative** to the ground truth rather than
as comparision to ground truth.

In this presentation the error above is hmer indel error rather than two
non-hmer indel errors.

The main function in this module is `reinterpret variants` that receives
the concordance dataframe (that contains both the variants from the
ground truth and the calls) and tries to compare the called  SNPs and
non-hmer indels to the variants in the ground truth to identify the differences
that can be explained by a small number of hmer indel errors.
"""
from __future__ import annotations

import itertools
import re
from typing import Any  # , List, Optional, Tuple

import numpy as np
import pandas as pd
import pyfaidx

import ugbio_core.flow_based_read as fbr
import ugvc.utils.misc_utils as utils
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugvc.vcfbed import vcftools


def reinterpret_variants(concordance_df: pd.DataFrame, in_dict: dict, fasta: pyfaidx.Fasta) -> pd.DataFrame:
    """Reinterprets the variants by comparing each called variant in the `concordance` dataframe
    to the ground truth in flow space and each false negative variant to the calls in flow space.
    If the sets differ by a single hmer indel, the call's definition is changed to hmer indel.

    Moreover, if the false negative is a result of a false positive hmer indel (i.e. if AT is called as ATT),
    we convert this call to true positive.

    Parameters
    ----------
    concordance_df: pd.DataFrame
        Input dataframe
    fasta: pyfaidx.Fasta
        Indexed FASTA
    in_dict : dict
        Dictionary with the following required keys:
         - `pos-gtr`, `pos_fps`, 'pos_fns' - positions of false positives, ground truth
         and false negative variants **on a single chromosome**. (np.ndarray)
         - dataframes `ugi`, `gtr`, `fps`, `fns` of Call variants, Ground truth
        variants, false positives and false negatives. see
        `vcf_pipeline_utils._get_locations_to_work_on` for generation example.

    Returns
    -------
    pd.DataFrame
        Concordance dataframe with false positives reinterpreted by
        comparison to the ground truth. I.e. call ATTTT versus ATTT
        is not a false positive non-hmer indel, rather false positive
        3-mer hmer indel. Similarly false negatives like this are
        converted to the true positive.


    """

    # Generate region that spans variants around each false positive variant
    variant_intervals = [vcftools.get_region_around_variant(x, in_dict["pos_gtr"]) for x in in_dict["pos_fps"]]

    # compare haplotypes containing each false positive variant with all haplotypes
    # that the ground truth variants around it can generate
    # Find two haplotypes that are the closest in flow space
    corrections_fps, best_pairs = compare_two_sets_of_variants(
        in_dict["pos_fps"],
        in_dict["ugi"],
        in_dict["gtr"],
        fasta,
        "gt_ultima",
        "gt_ground_truth",
        variant_intervals,
    )
    # Convert false positives for which there is a difference of a single
    # flow to hmer indels
    concordance_df = _apply_corrections(concordance_df, in_dict["fps"].index, corrections_fps)

    # compare haplotypes containing each false negative variant with all haplotypes
    # that the calls around it can generate
    # Find two haplotypes that are the closest in flow space
    corrections_fns, _ = compare_two_sets_of_variants(
        in_dict["pos_fns"],
        in_dict["gtr"],
        in_dict["ugi"],
        fasta,
        "gt_ground_truth",
        "gt_ultima",
        variant_intervals,
        best_pairs,
    )

    # Convert false negatives for which there is a difference of a single
    # flow to hmer indels.

    concordance_df = _apply_corrections(concordance_df, in_dict["fns"].index, corrections_fns)

    # we classify 1->0 deletions as non-hmer indels and thus we need to
    # fix their hmer indel length to 0
    concordance_df = _fix_zero_mer_indels(concordance_df)

    # copy correction fields to hmer_indel_length field
    concordance_df = _mark_hmer_indels(concordance_df)

    # If the corresponding pair of
    # haplotypes was counted in false positives - convert the false negative
    # to the true positive. (For these cases corrections_fns=(-1,-1))
    concordance_df = _convert_fns_to_tps(concordance_df)
    return concordance_df


def compare_two_sets_of_variants(
    positions_to_test: list,
    set1_variants: pd.DataFrame,
    set2_variants: pd.DataFrame,
    fasta: pyfaidx.Fasta,
    set1_genotype_col: str,
    set2_genotype_col: str,
    variant_intervals: list,
    given_haplotype_pairs: set | None = None,
) -> tuple[list, set]:
    """Asks how many hmer indels are needed to convert variants in set1 to variant in set2

    The function receives a list of variant positions in `positions_to_test, and two
    sets of variants: `set1` and `set2`. It generates all haplotypes from the variants around
    each positions in set1, all haplotypes around the same positions in set2.

    The haplotypes are being generated on the `variant_interval` that contains the position.


    It then looks for the pair of haplotypes from the two sets that have a minimal number of
    flow differences and returns the best difference in the same format as in `compare_haplotypes`
    For every location it finds list of pairs of the closest haplotypes

    Finally if the pair of the haplotypes is contained in `given_haplotype_pairs`, the correction
    will be (-1,None) meaning that the variant should be requalified as "tp"

    Parameters
    ----------
    positions_to_test : list
        List of positions of the variants to test
    set1_variants : pd.DataFrame
        The first set of variants (e.g. Ultima call set)
    set2_variants : pd.DataFrame
        The second set of variants (e.g. Ground truth)
    fasta : pyfaidx.Fasta
        Description
    set1_genotype_col : str
        Genotype column of set1_variants
    set2_genotype_col : str
        Genotype column of set2_variants
    variant_intervals: list
        List of intervals for generation of the haplotypes
    given_haplotype_pairs: list
        list of tuples of the previously found best pairs of haplotypes

    Returns
    -------
    tuple
        (list, list) - list of best differences and list of lists of pairs of the
        closest haplotypes in flow space

    See also
    --------


    """
    if given_haplotype_pairs is None:
        given_haplotype_pairs = set()

    chromosome = set(set1_variants["chrom"])
    if len(chromosome) == 0:
        return [], set()
    assert len(chromosome) == 1, "All variants should belong to a single chromosome"
    chromosome = chromosome.pop()

    interval_starts = [x[0] for x in variant_intervals]
    corrections: list[tuple[int, Any]] = []
    haplotype_results: list[tuple] = []

    # Go over all positions
    for pos in positions_to_test:

        # Find the interval in which the position is included
        select_region = _select_best_region(interval_starts, variant_intervals, pos)
        if select_region is None:
            # there is no interval in which the variant belongs,
            # this means that it should not be corrected. This can happen
            # e.g. when correcting false negatives for which no close
            # call exists

            corrections.append((100, None))
            continue

        # Get all variants in both sets from the region
        variants_set2 = vcftools.get_variants_from_region(variant_df=set2_variants, region=select_region)
        reference = get_reference_from_region(refidx=fasta[chromosome], region=select_region)

        # Generate sets of haplotypes
        haplotypes_set2 = apply_variants_to_reference(
            reference,
            variants_set2,
            select_region[0],
            set2_genotype_col,
            include_ref=True,
            exclude_ref_pos=None,
        )

        # Convert haplotypes to flows
        flows_set2 = [fbr.generate_key_from_sequence(x, DEFAULT_FLOW_ORDER) for x in haplotypes_set2]

        variants_set1 = vcftools.get_variants_from_region(variant_df=set1_variants, region=select_region)

        haplotypes_set1 = apply_variants_to_reference(
            reference,
            variants_set1,
            select_region[0],
            set1_genotype_col,
            include_ref=False,
            exclude_ref_pos=pos,
        )

        flows_set1 = [fbr.generate_key_from_sequence(x, DEFAULT_FLOW_ORDER) for x in haplotypes_set1]

        # find the minimal distance between pair of haplotypes
        correction = compare_haplotypes(flows_set2, flows_set1)

        # Find all haplotype pairs that have this distance
        best_haplotypes = find_best_pairs_of_haplotypes(flows_set2, flows_set1, correction)

        # check if this pair of haplotypes was already encountered
        # if so - we already counted this error (report (-1,-1))
        haplotype_results += best_haplotypes
        for best_hap in best_haplotypes:
            if best_hap in given_haplotype_pairs:
                correction = (-1, -1)
            break

        corrections.append(correction)

    haplotype_results_set = set(haplotype_results)
    return corrections, haplotype_results_set


def apply_variants_to_reference(
    reference: str,
    variants: pd.DataFrame,
    offset: int,
    genotype_col: str,
    include_ref: bool,
    exclude_ref_pos: int | None = None,
) -> list:
    """Collects a reference subsequence, and a list of variants and returns all
    possible haplotypes that can be obtained by applying these variants to the sequence

    Parameters
    ----------
    reference : str
        Reference sequence
    variants : pd.DataFrame
        Varaints (dataframe) assumed to be all contained in the reference sequence.
        Variants dataframe should contain the following columns: `pos, genotype_col, alleles`
    offset : int
        Genomic position of the zeroth base of the reference sequence
    genotype_col : str
        The column in the dataframe that contains the called genotype of the variant. Only alleles
        mentioned in the genotype_col are used in the haplotypes. (See `include_ref`)
    include_ref : bool
        Should reference allele (0) be added in a addition to the alleles in the `genotype_col`
    exclude_ref_pos : int, optional
        variant position where reference allele should never be used (when only haplotypes
        with non-reference alleles are required)

    Returns
    -------
    list
        List of strings - haplotype sequences
    """

    variant_positions = list(variants.pos - offset)

    # Break the reference sequence on the locatoins of the variants
    chunks = [reference[i:j] for i, j in zip([0] + variant_positions, variant_positions + [len(reference)])]

    rls = np.array(variants.ref.apply(len))
    # remove the reference allele sequence from the beginning of each chunk
    chunks = [chunks[0]] + [x[0][x[1] :] for x in zip(chunks[1:], rls)]

    # generate sets of alleles for each positions according to the genotypes
    # for each position
    alleles = list(variants["alleles"])
    gts = list(variants[genotype_col])
    pos = list(variants.pos)

    # add the reference allele to the alleles tested in addition to the
    # genotype
    if include_ref:
        for i, this_gt in enumerate(gts):
            if 0 in this_gt:
                continue
            gts[i] = (0,) + this_gt

    # never add reference allele (when testing the variant)
    if exclude_ref_pos is not None:
        for i, (this_gt, this_pos) in enumerate(zip(gts, pos)):
            if this_pos == exclude_ref_pos:
                gts[i] = tuple(x for x in this_gt if x != 0)

    gts = [[y for y in x if y is not None] for x in gts]

    gts = [sorted(x) for x in gts]
    alleles = [[alleles[i][n] for n in gts[i]] for i in range(len(alleles))]
    alleles = [[y for y in x if not y.startswith("<") and "*" not in y] for x in alleles]

    results = []
    for combination in itertools.product(*alleles):
        results.append(_apply_alleles_to_ref(chunks, combination))
    return results


def compare_haplotypes(flows: list, flows_variant: list) -> tuple[int, Any]:
    """Compare two lists of flow-based haplotypes (ground truth and calls), finds the closest pair
    and returns the number of differences and the length of the **ground truth** hmer(s) that were
    different

    Parameters
    ----------
    flows : list
        List of flow-based haplotypes (from ground truth)
    flows_variant : list
        List of flow-based haplotypes (from calls)

    Returns
    -------
    tuple
        (int, Union[int, list, None]). The minimal distance (in flow-space) betweeh the
        haplotype in the first and the second list.
        The first value is the number of different flows (1,2 or >2 is reported as 100)
        If the first value is 100 - the second value is None
        If the first value is 1 - the second value is the value of the hmer that is different
        _in the ground truth_ haplotype.
        If the first value is 2 - the second value is a list of the lengths of the hmer
        in the ground truth haplotype

    Examples
    --------
     List1=[[1,0,1], [0,1,1]]
     List2=[[1,1,1], [2,1]]

    The function returns (1,0): the best pair has a single different
    hmer, that is 0 in list1

     List1=[[1,0,1], [0,1,1]]
     List2=[[1,1], [2,1]]

    The functions returns (100, None): There is no pair of the same
    length between list1 and list2

     List1=[[1,1,0], [0,1,1]]
     List2=[[1,0,1],[2,1]]

    The function returns (2,[1,0]): There is a difference in two hmers: 0 and 1
    """

    best_diff = (100, None)

    for gtr_flow in flows:
        for var_flow in flows_variant:

            # if the haplotypes are of different length in flow - their difference
            # is for sure not a single hmer
            if len(gtr_flow) != len(var_flow):
                continue

            diff_flow = gtr_flow - var_flow
            if (diff_flow != 0).sum() <= best_diff[0]:
                cur_best_n_changes, cur_best_hmer = best_diff
                best_diff = ((diff_flow != 0).sum(), None)

                # we update the best result if it was more than a single change
                # and now we have a single change haplotype pair, or if in this pair
                # the different homopolymer is higher
                # TODO: better logic for two different hmers.
                if best_diff[0] == 1:
                    if cur_best_n_changes != 1:
                        best_diff = (1, gtr_flow[diff_flow != 0][0])
                    elif cur_best_n_changes == 1 and gtr_flow[diff_flow != 0][0] > cur_best_hmer:
                        best_diff = (1, gtr_flow[diff_flow != 0][0])
                    else:
                        # do not update the result
                        best_diff = (cur_best_n_changes, cur_best_hmer)

                if best_diff[0] == 2:
                    best_diff = (2, gtr_flow[diff_flow != 0])
    return best_diff


def find_best_pairs_of_haplotypes(flows: list, flows_variant: list, best_diff: tuple) -> list:
    """Finds Which pairs of haplotypes in the two sets differ by a single hmer
    with a specified value in the first set. `best_diff` is the result of `compare_haplotypes`
    on the two sets and it is expected that `best_diff[0]=1`.

    Parameters
    ----------
    flows : list
        Flow-space haplotypes of the first set
    flows_variant : list
        Flow-space haplotypes of the second set
    best_diff : tuple
        The best difference between the haplotypes (output of `compare_haplotypes`)

    Returns
    -------
    list
        List of pairs of haplotypes that satisfy `best_diff`

    See also
    --------
    compare_haplotypes
    """

    # we only care about haplotypes that differ in a single hmer
    if best_diff[0] > 1:
        return []

    best_pairs = []
    for gtr_flow in flows:
        for var_flow in flows_variant:
            # if the two haplotypes are of different length they
            # do not differ by a single hmer
            if len(gtr_flow) != len(var_flow):
                continue

            diff_flow = gtr_flow - var_flow
            if (diff_flow != 0).sum() == 1:
                # if the two haplotypes differ by a single hmer with
                # value equal to best_diff[1] - add to the results
                if gtr_flow[diff_flow != 0][0] == best_diff[1]:
                    best_pairs.append(tuple(sorted((gtr_flow.tostring(), var_flow.tostring()))))
    return best_pairs


def _apply_alleles_to_ref(chunks: list, combination: list | tuple) -> str:
    "Concatenates reference sequences with allele sequences"

    return "".join([x[0] + x[1] for x in itertools.zip_longest(chunks, combination, fillvalue="")])


def _convert_fns_to_tps(_df: pd.DataFrame) -> pd.DataFrame:
    "Converts false negative calls to true positive if"

    change_annotation = _df["compare_to_gtr_changes"] == -1
    subdf = _df.loc[change_annotation, ["indel", "classify", "hmer_indel_length", "qual", "sor"]].copy()
    subdf["classify"] = "tp"
    subdf["qual"] = 300
    subdf["sor"] = 1
    _df.loc[subdf.index, subdf.columns] = subdf
    _df.loc[pd.isnull(_df["qual"]), "qual"] = 50
    return _df


def _mark_hmer_indels(_df: pd.DataFrame) -> pd.DataFrame:
    """Sets re-interpreted hmer indels. Variants that are hmer indels
    relative to the ground truth are set as hmer indels in the fields"""
    hmer_indels = _df.compare_to_gtr_changes == 1
    _df.loc[hmer_indels, "indel"] = True
    _df.loc[hmer_indels, "hmer_indel_length"] = _df.loc[hmer_indels, "compare_to_gtr_hmer_indel_len"]
    return _df


def _apply_corrections(_df: pd.DataFrame, positions: pd.Index, corrections: list) -> pd.DataFrame:
    """Adds the result flow-based haplotype comparison to the dataframe in fields
    `compare_to_gtr_changes`: number of different flows and
    `compare_to_gtr_hmer_indel_len`: length of the gtr indel(s) that is (are) different
    """

    compare_to_gtr_result = [x[0] if x[0] < 3 else 100 for x in corrections]
    hmer_indel_to_gtr = [x[1] if x[0] < 3 else 0 for x in corrections]
    if len(positions) == 1:  # solves a boundary case where there is a single position and hmer_indel_to_gtr is an array
        _df.at[positions[0:1], "compare_to_gtr_hmer_indel_len"] = hmer_indel_to_gtr[0]
        _df["compare_to_gtr_hmer_indel_len"] = _df["compare_to_gtr_hmer_indel_len"].astype(object)
    else:
        _df.loc[positions, "compare_to_gtr_changes"] = compare_to_gtr_result
        _df.loc[positions, "compare_to_gtr_hmer_indel_len"] = pd.Series(hmer_indel_to_gtr, index=positions)
    return _df


def _fix_zero_mer_indels(_df: pd.DataFrame) -> pd.DataFrame:
    """Fixes 1->0 deletions that corrections show as 1-mer."""
    take = (
        (_df["indel_classify"] == "del")
        & (_df["compare_to_gtr_changes"] == 1)
        & (_df["compare_to_gtr_hmer_indel_len"].apply(lambda x: (not isinstance(x, np.ndarray)) and (x == 1)))
    )
    _df.loc[take, "compare_to_gtr_hmer_indel_len"] = 0
    return _df


def get_reference_from_region(refidx: pyfaidx.FastaRecord | str, region: tuple) -> str:
    """Get the reference sequence of the region

    Parameters
    ----------

    refidx : pyfaidx.FastaRecord or str
        Indexed fasta (`pyfaidx.Fasta`) - single chromosome
    region : tuple
        [start, end), 1-based coordinates of the region!
    Returns
    -------
    str
        Reference string (uppercase, Ns and other non-standard characters replaced with As)
    """
    unsanitized_ref = str(refidx[region[0] - 1 : region[1] - 1]).upper()
    return re.sub(r"([^.ATCG])", "A", unsanitized_ref)


def _select_best_region(interval_starts: list, intervals: list, pos: int) -> tuple | None:
    """Out of list of intervals given by interval_starts and intervals (for speed two varianbles),
    found the interval with the highest distance of pos from the ends.
    """

    pos1 = np.searchsorted(interval_starts, pos).astype(int) - 1
    best_distance = None
    best_index = -1
    while pos1 >= 0 and utils.isin(pos, intervals[pos1]):
        dist = min(pos - intervals[pos1][0], intervals[pos1][1] - pos)
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_index = pos1
        pos1 -= 1
    # if no interval found that contains pos
    if best_distance is None:
        return None

    # if found an interval
    return intervals[best_index]
