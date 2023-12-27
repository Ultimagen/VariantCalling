from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyfaidx
import tqdm.auto as tqdm

import ugvc.comparison.vcf_pipeline_utils as vpu
import ugvc.filtering.multiallelics as mu
import ugvc.filtering.spandel as sp
from ugvc import logger
from ugvc.vcfbed import vcftools

IGNORE = -1
MISS = -2
warnings.simplefilter(action="ignore", category=FutureWarning)


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
        f"Counts of 'BASE' tag of the variants that were in the vcfeval output but \
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


def prepare_ground_truth(input_vcf, base_vcf, hcr, reference, output_h5, chromosome: list = None):
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
        labeled_df = process_multiallelic_spandel(labeled_df, reference, chrom, input_vcf)
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


def process_multiallelic_spandel(df: pd.DataFrame, reference: str, chromosome: str, vcf: str) -> pd.DataFrame:
    """Process multiallelic variants and spanning deletions

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    reference : str
        Reference FASTA
    chromosome : str
        Chromosome to process
    vcf : str
        VCF file of the callset (to extract the header)

    Returns
    -------
    pd.DataFrame
        Re-processed dataframe, multiallelic variants replaced by multiple biallelic rows
        Spanning deletion variants replaced
        The corresponding row of the dataframe is marked by "multiallelic_group column"
    """
    df = df.copy()
    overlaps = mu.select_overlapping_variants(df)
    combined_overlaps = sum(overlaps, [])
    multiallelics = [x for x in overlaps if len(x) == 1]
    spandels = [x for x in overlaps if len(x) > 1]
    fasta = pyfaidx.Fasta(reference)
    multiallelic_groups = [
        mu.split_multiallelic_variants(df.iloc[olp[0]], vcf, fasta[chromosome]) for olp in tqdm.tqdm(multiallelics)
    ]
    for n in multiallelic_groups:
        n.loc[:, "multiallelic_group"] = [(n.iloc[0]["chrom"], n.iloc[0]["pos"])] * (n.shape[0])
    multiallelic_groups = pd.concat(multiallelic_groups, ignore_index=True)
    multiallelic_groups = mu.cleanup_multiallelics(multiallelic_groups)
    spanning_deletions = [
        pd.concat(
            [df.iloc[olp[0] : olp[0] + 1]]
            + [
                sp.split_multiallelic_variants_with_spandel(df.iloc[olp[i]], df.iloc[olp[0]], vcf, fasta[chromosome])
                for i in range(1, len(olp))
            ]
        )
        for olp in tqdm.tqdm(spandels)
    ]

    for i, n in enumerate(spanning_deletions):
        n.loc[:, "spanning_deletion"] = [(df.iloc[spandels[i][0]]["chrom"], df.iloc[spandels[i][0]]["pos"])] * (
            n.shape[0]
        )
    spanning_deletions = pd.concat(spanning_deletions, ignore_index=True)
    spanning_deletions = mu.cleanup_multiallelics(spanning_deletions)

    idx_multi_spandels = df.index[combined_overlaps]
    df.drop(idx_multi_spandels, axis=0, inplace=True)
    mug = pd.concat((multiallelic_groups, spanning_deletions), ignore_index=True)  # resetting index
    return pd.concat((df, mug))
