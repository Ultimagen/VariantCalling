'''Module for flow-based reinterpretation of variants

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
'''

import numpy as np
import pandas as pd
import pyfaidx
import itertools
from typing import Optional
import python.vcftools as vcftools
import python.utils as utils


def get_reference_from_region(refidx: pyfaidx.Fasta, region: tuple) -> str:
    """Get the reference sequence of the region

    Parameters
    ----------
    refidx : pyfaidx.Fasta
        Indexed fasta (`pyfaidx.Fasta`) - single chromosome
    region : tuple
        (start, end)

    Returns
    -------
    str
        Reference string (uppercase, Ns replaced with As)
    """
    return str(refidx[region[0] - 1: region[1] - 1]).upper().replace("N", "A")


def apply_variants_to_reference(
    reference: str,
    variants: pd.DataFrame,
    offset: int,
    genotype_col: str,
    include_ref: bool,
    exclude_ref_pos: Optional[int] = None,
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
    exclude_ref_pos : Optional[int], optional
        variant position where reference allele should never be used (when only haplotypes
        with non-reference alleles are required)

    Returns
    -------
    list
        List of strings - haplotype sequences
    """

    variant_positions = list(variants.pos - offset)
    chunks = [
        reference[i:j]
        for i, j in zip([0] + variant_positions, variant_positions + [len(reference)])
    ]
    rls = np.array(variants.ref.apply(len))
    chunks = [chunks[0]] + [x[0][x[1]:] for x in zip(chunks[1:], rls)]

    alleles = list(variants["alleles"])
    gts = list(variants[genotype_col])
    pos = list(variants.pos)
    if include_ref:
        for i in range(len(gts)):
            if 0 in gts[i]:
                continue
            else:
                gts[i] = (0,) + gts[i]
    if exclude_ref_pos is not None:
        for i in range(len(gts)):
            if pos[i] == exclude_ref_pos:
                gts[i] = tuple([x for x in gts[i] if x != 0])
    gts = [[y for y in x if y is not None] for x in gts]

    gts = [sorted(x) for x in gts]
    alleles = [[alleles[i][n] for n in gts[i]] for i in range(len(alleles))]
    alleles = [
        [y for y in x if not y.startswith("<") and not "*" in y] for x in alleles
    ]
    results = []
    for combination in itertools.product(*alleles):
        results.append(_apply_alleles_to_ref(chunks, combination))
    return results


def compare_haplotypes(flows: list, flows_variant: list) -> tuple:
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
    """

    best_diff = (100, None)
    for sf in flows:
        for df in flows_variant:
            if len(sf) != len(df):
                continue
            else:
                d = sf - df
                if (d != 0).sum() <= best_diff[0]:
                    best_diff = ((d != 0).sum(), None)
                    cur_bd = best_diff[1]
                    cur_diff = best_diff[0]

                    if best_diff[0] == 1:
                        if cur_diff != 1:
                            best_diff = (1, sf[d != 0][0])
                        elif cur_diff == 1 and (
                            cur_bd is None or sf[d != 0][0] > cur_bd
                        ):
                            best_diff = (1, sf[d != 0][0])
                    if best_diff[0] == 2:
                        best_diff = (2, sf[d != 0])
    return best_diff


def find_best_pairs_of_haplotypes(
    flows: list, flows_variant: list, best_diff: tuple
) -> list:
    """Given the two list of haplotypes (in flow space) and the results of the closest ones, 
    returns all pairs that fulfill this difference

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
    if best_diff[0] > 1:
        return []
    else:
        best_pairs = []
        for sf in flows:
            for df in flows_variant:
                if len(sf) != len(df):
                    continue
                else:
                    d = sf - df
                    if (d != 0).sum() == 1:
                        if sf[d != 0][0] == best_diff[1]:
                            best_pairs.append(
                                tuple(sorted((sf.tostring(), df.tostring())))
                            )
        return best_pairs


def compare_two_sets_of_variants(positions_to_test: list,
                                 set1_variants: pd.DataFrame,
                                 set2_variants: pd.DataFrame,
                                 fasta: pyfaidx.Fasta,
                                 set1_genotype_col: str,
                                 set2_genotype_col: str,
                                 ) -> tuple:
    '''Asks how many hmer indels are needed to convert variant in set1 to variant in set2

    The function receives a list of variant positions in `positions_to_test, and two 
    sets of variants: `set1` and `set2`. It generates all haplotypes from the variants around 
    each positions in set1, all haplotypes around the same positions in set2. 

    It then looks for the pair of haplotypes from the two sets that have a minimal number of 
    flow differences and returns the best difference in the same format as in `compare_haplotypes`
    Finally, for every location it finds list of pairs of the closest haplotypes 

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

    Returns
    -------
    tuple
        (list, list) - list of best differences and list of lists of pairs of the 
        closest haplotypes in flow space

    '''
    chromosome = set(set1_variants['chrom'])
    assert len(chromosome)==1, "All variants should belong to a single chromosome"
    chromosome = chromosome.pop()

    positions_other_set = np.array(set2_variants.pos)
    vv = [vcftools.get_region_around_variant(x, positions_other_set)
          for x in positions_to_test]
    corrections = []
    best_pairs = []
    for i in range(len(vv)):
        variants_set2 = vcftools.get_variants_from_region(
            variant_df=set2_variants, region=vv[i])
        if variants_set2.shape[0] != 0:
            assert (variants_set2.pos.min() - vv[i][0] >= 10) & (
                vv[i][1] - variants_set2.pos.max() >= 10
            ), "Assertion failed"
        reference = get_reference_from_region(
            refidx=fasta[chromosome], region=vv[i])
        haplotypes_set2 = apply_variants_to_reference(
            reference,
            variants_set2,
            vv[i][0],
            set2_genotype_col,
            include_ref=True,
            exclude_ref_pos=None,
        )
        flows_set2 = [utils.generateKeyFromSequence(
            x, "TACG") for x in haplotypes_set2]
        variants_set1 = vcftools.get_variants_from_region(
            variant_df=set1_variants, region=vv[i])
        haplotypes_set1 = apply_variants_to_reference(
            reference,
            variants_set1,
            vv[i][0],
            set1_genotype_col,
            include_ref=False,
            exclude_ref_pos=positions_to_test[i],
        )
        flows_set1 = [utils.generateKeyFromSequence(
            x, "TACG") for x in haplotypes_set1]
        correction = compare_haplotypes(flows_set2, flows_set1)
        best_pairs += find_best_pairs_of_haplotypes(
            flows_set2, flows_set1, correction)
        corrections.append(correction)
    return corrections, best_pairs


def _apply_alleles_to_ref(chunks: list, combination: list) -> str:
    "Concatenates reference sequences with allele sequences"

    return "".join(ech
                   [x[0] + x[1]
                    for x in itertools.zip_longest(chunks, combination, fillvalue="")]
                   )
