from __future__ import annotations

import numpy as np
import pandas as pd
import pyfaidx
import tqdm.auto as tqdm

import ugvc.comparison.flow_based_concordance as fbc
import ugvc.comparison.vcf_pipeline_utils as vpu
import ugvc.flow_format.flow_based_read as fbr
from ugvc import logger
from ugvc.vcfbed import vcftools

IGNORE = -1
MISS = -2


def calculate_labeled_vcf(call_vcf: str, vcfeval_vcf: str, contig: str) -> pd.DataFrame:
    df_original = vcftools.get_vcf_df(call_vcf, chromosome=contig)
    df_vcfeval = vcftools.get_vcf_df(vcfeval_vcf, chromosome=contig)

    joint_df = df_original.join(df_vcfeval, rsuffix="_vcfeval")
    missing = df_vcfeval.loc[df_vcfeval.index.difference(joint_df.index)]
    counts_base = missing["base"].value_counts()
    counts_call = missing["call"].value_counts()
    logger.info(f"Duplicates in df_original: {df_original.index.duplicated().sum()}")
    logger.info(f"Duplicates in vcfeval: {df_vcfeval.index.duplicated().sum()}")

    logger.info(
        f"Counts of 'BASE' tag of thevariants that were in the vcfeval output but \
        did not match to call_vcf: {counts_base}"
    )
    logger.info(
        f"Counts of 'CALL' tag of the variants that were in the vcfeval output but \
            did not match to call_vcf: {counts_call}"
    )
    vcfeval_miss = (pd.isnull(joint_df["call"]) & (pd.isnull(joint_df["base"]))).sum()
    logger.info(f"Number of records in call_vcf that did not exist in vcfeval_vcf: {vcfeval_miss}")
    return joint_df


def calculate_labels(labeled_df: pd.DataFrame) -> pd.Series:
    result_gt1 = pd.Series(np.zeros(labeled_df.shape[0]), index=labeled_df.index)
    result_gt2 = pd.Series(np.zeros(labeled_df.shape[0]), index=labeled_df.index)
    nans = (
        (labeled_df["call"] == "OUT")
        | (labeled_df["call"] == "IGN")
        | (pd.isnull(labeled_df["call"]))
        | (labeled_df["call"] == "HARD")
    )
    result_gt1[nans] = IGNORE
    result_gt2[nans] = IGNORE
    tps = labeled_df["call"] == "TP"
    result_gt1[tps] = labeled_df.loc[tps, "gt"].apply(lambda x: x[0])
    result_gt2[tps] = labeled_df.loc[tps, "gt"].apply(lambda x: x[1])

    fps = labeled_df["call"] == "FP"
    result_gt1[fps] = 0
    result_gt2[fps] = 0

    fp_ca_df = labeled_df[labeled_df["call"] == "FP_CA"]  # these are either genotyping errors or allele misses
    if fp_ca_df.shape[0] > 0:
        alleles_calls = fp_ca_df["alleles"]

        def soft_index(x, y):
            try:
                return x.index(y)
            except ValueError:
                return MISS

        # this is to fix the cases where gt genotype is (None,1) or (None,) or something else
        fp_ca_df["gt_vcfeval"].where(
            fp_ca_df["gt_vcfeval"].apply(lambda x: None not in x and len(x) == 2),
            pd.Series([(0, 0)] * len(fp_ca_df["gt_vcfeval"]), index=fp_ca_df.index),
            inplace=True,
        )

        alleles_used_vcfeval = fp_ca_df.apply(
            lambda x: (x["alleles_vcfeval"][x["gt_vcfeval"][0]], x["alleles_vcfeval"][x["gt_vcfeval"][1]]), axis=1
        )
        tmp = pd.DataFrame({"alleles_calls": alleles_calls, "alleles_used_vcfeval": alleles_used_vcfeval})
        gt_genotypes = tmp.apply(
            lambda x: (
                soft_index(x["alleles_calls"], x["alleles_used_vcfeval"][0]),
                soft_index(x["alleles_calls"], x["alleles_used_vcfeval"][1]),
            ),
            axis=1,
        )

        result_gt1[fp_ca_df.index] = gt_genotypes.apply(lambda x: x[0])
        result_gt2[fp_ca_df.index] = gt_genotypes.apply(lambda x: x[1])
    result = pd.DataFrame({0: result_gt1.astype(int), 1: result_gt2.astype(int)})
    result["true_gt"] = result.apply(lambda x: (x[0], x[1]), axis=1)
    return result["true_gt"]


def prepare_ground_truth(input_vcf, base_vcf, hcr, reference, output_h5, chromosome=None):
    pipeline = vpu.VcfPipelineUtils()
    vcfeval_output = pipeline.run_vcfeval_concordance(
        input_file=input_vcf,
        truth_file=base_vcf,
        output_prefix=input_vcf.replace(".vcf.gz", ""),
        ref_genome=reference,
        evaluation_regions=hcr,
        ignore_filter=True,
    )
    if chromosome is None:
        chromosome = [f"chr{x}" for x in list(range(1, 23)) + ["X", "Y"]]

    for chrom in tqdm.tqdm(chromosome):
        labeled_df = calculate_labeled_vcf(input_vcf, vcfeval_output, contig=chrom)
        labels = calculate_labels(labeled_df)
        labeled_df["label"] = labels
        labeled_df.to_hdf(output_h5, key=chrom, mode="a")


def encode_labels(ll):
    return [encode_label(x) for x in ll]


def encode_label(label):

    label = tuple(sorted(label))
    if label == (0, 0):
        return 2
    if label == (0, 1):
        return 0
    if label == (1, 1):
        return 1
    raise RuntimeError(f"Encoding of gt={label} not supported")


def decode_label(label):
    decode_dct = {0: (0, 1), 1: (1, 1), 2: (0, 0)}
    return decode_dct[label]


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


def split_multiallelic_variants(multiallelic_variant: pd.Series) -> pd.DataFrame:
    """Splits multiallelic variants into multiple rows

    Parameters
    ----------
    multiallelic_variant : pd.Series
        A row from the training set dataframe
    Returns
    -------
    pd.DataFrame
        A dataframe with the same columns as the training set dataframe
    """
    # record_to_nbr_dict = vcftools.header_record_number(call_vcf_header)
    alleles = multiallelic_variant["alleles"]
    pos = multiallelic_variant["pos"]
    ref = multiallelic_variant["ref"]
    alt = multiallelic_variant["alt"]
    result = pd.DataFrame()
    for i, a in enumerate(alleles):
        if a == ref:
            continue
        if a == alt:
            continue
        result = result.append(multiallelic_variant)
        result.iloc[-1]["alleles"] = [ref, a]
        result.iloc[-1]["pos"] = pos + i
    return result


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
