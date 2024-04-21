from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import pysam
import tqdm.auto as tqdm

import ugvc.comparison.vcf_pipeline_utils as vpu
import ugvc.filtering.multiallelics as mu
import ugvc.filtering.spandel as sp
from ugvc import logger
from ugvc.filtering.tprep_constants import IGNORE, MISS
from ugvc.vcfbed import vcftools

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def calculate_labeled_vcf(
    call_vcf: str, vcfeval_vcf: str, contig: str, custom_info_fields: list[str] | None = None
) -> pd.DataFrame:
    """Receives a VCF and a result of its comparison (vcfeval) and returns a joint vcf ready to create training labels.
    The function also prints some statistics about the join, to help debugging.

    Parameters
    ----------
    call_vcf : str
        Call VCF
    vcfeval_vcf : str
        VCFEVAL VCF (vcfeval should run in 'combine' mode)
    contig : str
        Chromosome to work on
    custom_info_fields: list, optional
        List of custom INFO fields to read from the VCF
    Returns
    -------
    pd.DataFrame
        Joined VCF, vcfeval coluns will get a suffix _vcfeval
    """
    df_original = vcftools.get_vcf_df(call_vcf, chromosome=contig, custom_info_fields=custom_info_fields)
    df_vcfeval = vcftools.get_vcf_df(vcfeval_vcf, chromosome=contig)
    if df_vcfeval.shape[0] == 0:
        logger.warning(f"No records in vcfeval_vcf for chromosome {contig}")
        return pd.DataFrame()
    if df_original.shape[0] == 0:
        logger.warning(f"No records in call_vcf for chromosome {contig}")
        return pd.DataFrame()
    joint_df = df_original.join(df_vcfeval, rsuffix="_vcfeval")
    missing = df_vcfeval.loc[df_vcfeval.index.difference(joint_df.index)]
    counts_base = missing["base"].value_counts()
    counts_call = missing["call"].value_counts()
    logger.info(f"Duplicates in df_original: {df_original.index.duplicated().sum()}")
    logger.info(f"Duplicates in vcfeval: {df_vcfeval.index.duplicated().sum()}")

    logger.info(
        "Counts of 'BASE' tag of the variants that were in the vcfeval output but "
        f"did not match to call_vcf:\n {counts_base}"
    )
    logger.info(
        "Counts of 'CALL' tag of the variants that were in the vcfeval output but "
        f"did not match to call_vcf:\n {counts_call}"
    )
    vcfeval_miss = (pd.isnull(joint_df["call"]) & (pd.isnull(joint_df["base"]))).sum()
    logger.info(f"Number of records in call_vcf that did not exist in vcfeval_vcf: {vcfeval_miss}")
    return joint_df


def calculate_labels(labeled_df: pd.DataFrame) -> pd.Series:
    """Receives a dataframe that is output of calculate_labeled_vcf and returns a series of label for each variant.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Joint dataframe

    Returns
    -------
    pd.Series
        Series of tuples for the genotype for each variant. Note that -1 and -2 are IGNORE and MISS values
        They are used to mark false negatives or variants outside of the HCR region. IGNORE is for variants
        that are labeled by vcfeval as OUT/IGN/HARD, MISS are for alleles that were missing from the callset
    """

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


def prepare_ground_truth(
    input_vcf: str,
    base_vcf: str,
    hcr: str,
    reference: str,
    output_h5: str,
    chromosome: list | None = None,
    test_split: str | None = None,
    custom_info_fields: list[str] | None = None,
    ignore_genotype: bool = False,
) -> None:
    """Generates a training set dataframe from callset and the ground truth VCF. The following steps are peformed:
    1. Run vcfeval to compare the callset to the ground truth
    2. Join the callset and the vcfeval output
    3. Calculate the labels for each variant
    4. Process multiallelic variants and spanning deletions (splitting them into multiple lines,
    asking each time if one or two non-reference alleles are present). For example if the true
    variant is (1,2), we will generate two rows in the training df:
    in the first row we will have the reference and the strongest
    alt allele and the label is (1,1) and in the second row we will have
    the strongest alt allele and the second strongest alt allele and the label will
    be (0, 1)
    If test_train_split is a name of a chromosome, that chromosome is saved in
    The results are saved in output_h5 file, with each chromosome in a separate key.

    Parameters
    ----------
    input_vcf : str
        Input (call) VCF file
    base_vcf : str
        Truth VCF file (e.g GIAB callset)
    hcr : str
        HCR bed file to generate training set on
    reference : str
        Reference fasta. Note that .sdf file will be required
    output_h5 : str
        Output file
    chromosome : list, optional
        List of chromosomes to operate on, by default None
    test_split : str, optional
        The test set will be either single chromosome (str, will be saved in a separate file)
        or None (in which case no test set is produced)
    custom_info_fields: list, optional
        List of custom INFO annotations to read
    ignore_genotype: bool, optional
        Don't compare genotype information, only compare if allele is present in ground-truth

    """
    pipeline = vpu.VcfPipelineUtils()
    vcfeval_output = pipeline.run_vcfeval_concordance(
        input_file=input_vcf,
        truth_file=base_vcf,
        output_prefix=input_vcf.replace(".vcf.gz", ""),
        ref_genome=reference,
        evaluation_regions=hcr,
        ignore_filter=True,
        ignore_genotype=ignore_genotype,
    )
    if chromosome is None:
        chromosome = [f"chr{x}" for x in list(range(1, 23)) + ["X", "Y"]]
    for chrom in tqdm.tqdm(chromosome):
        labeled_df = calculate_labeled_vcf(
            input_vcf, vcfeval_output, contig=chrom, custom_info_fields=custom_info_fields
        )
        if labeled_df.shape[0] == 0:
            logger.warning(f"Skipping chromosome {chrom}: empty")
            continue
        labels = calculate_labels(labeled_df)
        labeled_df["label"] = labels
        labeled_df = process_multiallelic_spandel(labeled_df, reference, chrom, input_vcf)
        if test_split == chrom:
            dirname = Path(output_h5).parent
            stemname = Path(output_h5).stem
            labeled_df.to_hdf((dirname / Path(stemname + "_test.h5")), key=chrom, mode="a")
        else:
            labeled_df.to_hdf(output_h5, key=chrom, mode="a")


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
    column_dtypes = df.dtypes
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
    if len(spandels) > 0:
        spanning_deletions = [
            pd.concat(
                [_split_multiallelic_if_necessary(df.iloc[olp[0] : olp[0] + 1], vcf, fasta[chromosome])]
                + [
                    sp.split_multiallelic_variants_with_spandel(
                        df.iloc[olp[i]], df.iloc[olp[0]], vcf, fasta[chromosome]
                    )
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
        mug = pd.concat((multiallelic_groups, spanning_deletions), ignore_index=True)  # resetting index
    else:
        mug = multiallelic_groups.reset_index()
    idx_multi_spandels = df.index[combined_overlaps]
    df.drop(idx_multi_spandels, axis=0, inplace=True)
    return pd.concat((df, mug)).astype(column_dtypes)


def _split_multiallelic_if_necessary(
    multiallelic_variant: pd.DataFrame,
    call_vcf_header: pysam.VariantHeader | pysam.VariantFile | str,
    ref: pyfaidx.FastaRecord,
) -> pd.DataFrame:
    """Splits a variant into multiallelics if necessary or returns the original variant

    Parameters
    ----------
    multiallelic_variant : pd.Series
        A row from the training set dataframe
    call_vcf_header :
        Header of the VCF : pysam.VariantHeader | pysam.VariantFile | str
    ref : pyfaidx.Fasta
        Reference chromosome

    Returns
    -------
    pd.DataFrame
        Dataframe of split variant
    """
    assert multiallelic_variant.shape[0] == 1, "Should be a single row"
    alleles = multiallelic_variant["alleles"].values[0]
    if len(alleles) == 2:
        return multiallelic_variant
    return mu.split_multiallelic_variants(multiallelic_variant.iloc[0], call_vcf_header, ref)


def label_with_approximate_gt(
    vcf: str,
    blacklist: str,
    output_file: str,
    chromosomes_to_read: list | None = None,
    test_split: str | None = None,
    custom_info_fields: list | None = None,
) -> None:
    """Use approximate ground truth to generate labels. Specifically, all variants that belong to the blacklist are
    considered false positives, all variants that have "id" tag are considered true positives.
    The rest are considered "unknown" and are removed from the dataframe. Writes to `output_file`
    a dataframe with the labeled on "label", split on chromosomes. If `test_split` is not None, the chromosome
    with this name will be written separately in test.h5 dataframe

    Parameters
    ----------
    vcf : str
        VCF file
    blacklist : str
        Blacklist dataframe
    output_file : str
        Output labeled dataframe
    chromosomes_to_read : list, optional
        List of chromosomes to operate on, by default None
    test_split : str, optional
        Chromosome to put aside as test
    custom_info_fields: list, optional
        The names of interval annotations to read from the VCF
    """
    logger.info("Training data with approximate GT")
    blacklist_df = pd.read_hdf(blacklist, key="blacklist")
    logger.info("Finished reading blacklist")
    blacklist_df.index = pd.MultiIndex.from_tuples(blacklist_df.index)
    logger.debug("Coverted blacklist index to MultiIndex")

    if chromosomes_to_read is None:
        chromosomes_to_read = [f"chr{x}" for x in list(range(1, 23)) + ["X", "Y"]]
    logger.info(f"Reading VCF from {len(chromosomes_to_read)} chromosomes")
    for chromosome in tqdm.tqdm(chromosomes_to_read):
        if custom_info_fields is None:
            custom_info_fields = []
        df = vcftools.get_vcf_df(vcf, chromosome=chromosome, custom_info_fields=custom_info_fields)
        if df.shape[0] == 0:
            logger.warning(f"Skipping chromosome {chromosome}: empty")
            continue
        df = df.merge(blacklist_df, left_index=True, right_index=True, how="left")
        df["bl"].fillna(False, inplace=True)
        classify_clm = "label"

        df[classify_clm] = -1
        df.loc[df["bl"], classify_clm] = 0
        df.loc[~(df["id"].isna()), classify_clm] = 1
        df = df[df[classify_clm] != -1]

        df.drop("bl", axis=1, inplace=True)
        if chromosome == test_split:
            dirname = Path(output_file).parent
            stemname = Path(output_file).stem
            df.to_hdf((dirname / Path(stemname + "_test.h5")), key=chromosome, mode="a")
        else:
            df.to_hdf(output_file, key=chromosome, mode="a")
