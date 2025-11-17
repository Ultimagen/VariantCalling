from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pysam
import tqdm.auto as tqdm
import ugbio_core.vcf_utils as vpu
from ugbio_core.vcfbed import vcftools


def get_parental_vcf_df(maternal_vcfs: dict, paternal_vcfs: dict) -> pd.DataFrame:
    """Get dataframe of parental VCF

    Parameters
    ----------
    maternal_vcfs : dict
        dictionary with keys: names of the children and values: paths to the maternal VCFs
    paternal_vcfs : dict
        dictionary with keys: names of the children and values: paths to the paternal VCFs

    Returns
    -------
    pd.DataFrame
        Dataframe corresponding to the combined parental VCFs.
        The columns are of the form {child id}-mother, {child id}-father
    """
    maternal_dfs = [(x + "-mother", vcftools.get_vcf_df(maternal_vcfs[x])) for x in tqdm.tqdm(maternal_vcfs)]
    maternal_df = pd.concat(dict(maternal_dfs), axis=1)
    paternal_dfs = [(x + "-father", vcftools.get_vcf_df(paternal_vcfs[x])) for x in tqdm.tqdm(paternal_vcfs)]
    paternal_df = pd.concat(dict(paternal_dfs), axis=1)
    parent_df = pd.concat([maternal_df, paternal_df], axis=1)
    return parent_df


def add_parental_qualities_to_denovo_vcf(denovo_vcf: str, parental_vcf_df: pd.DataFrame) -> pd.DataFrame:
    """Add recalibrated quality (from child/parent calling to the denovo VCF

    Parameters
    ----------
    denovo_vcf : str
        path to the denovo VCF
    parental_vcf_df: pd.DataFrame
        Dataframe with the parental VCFs (from get_parental_vcf_df)
    Returns
    -------
    pd.DataFrame
        Dataframe with recalibrated qual (pair_qual) added to each denovo call in the denovo_vcf
        The pair qual is the minimal qual of the variant in the two child/parent vcfs
    """
    denovo_df = vcftools.get_vcf_df(denovo_vcf)
    denovo_df = pd.concat([denovo_df, parental_vcf_df.xs("qual", axis=1, level=1)], axis=1)
    df_denovo_exp = denovo_df.explode("hiconfdenovo").explode("loconfdenovo")
    df_denovo_exp["denovosample"] = np.where(
        pd.isnull(df_denovo_exp["hiconfdenovo"]), df_denovo_exp["loconfdenovo"], df_denovo_exp["hiconfdenovo"]
    )
    called_samples_mother = set(x[:-7] for x in df_denovo_exp.columns if x.endswith("mother"))
    called_samples_father = set(x[:-7] for x in df_denovo_exp.columns if x.endswith("father"))
    assert called_samples_mother == called_samples_father, "Mismatch between maternal and paternal samples"
    called_samples = called_samples_mother
    incalled = df_denovo_exp["denovosample"].apply(lambda x: x in called_samples)
    df_denovo_exp = df_denovo_exp.loc[incalled]
    assert (
        df_denovo_exp.shape[0] > 0
    ), "No denovo calls found in the VCF or no overlap between the de novo vcf and the somatic calls"
    df_denovo_exp["pair_qual"] = df_denovo_exp.apply(
        lambda x: min(x[x["denovosample"] + "-father"].fillna(0), x[x["denovosample"] + "-mother"].fillna(0)), axis=1
    )
    df_denovo_exp["pair_qual"].fillna(0, inplace=True)
    return df_denovo_exp


def write_recalibrated_vcf(denovo_vcf: str, output_vcf_name: str, recalibrated_denovo: pd.DataFrame) -> None:
    """Write recalibrated VCF

    Parameters
    ----------
    denovo_vcf : str
        path to the denovo VCF
    output_vcf_name : str
        path to the output VCF
    recalibrated_denovo: pd.DataFrame
        Dataframe with the recalibrated denovo calls (from add_paternal_qualities_to_denovo_vcf)
    """

    with pysam.VariantFile(denovo_vcf) as input_vcf:
        hdr = input_vcf.header
        hdr.info.add("DENOVO_QUAL", "1", "Float", "Pair quality (min of child/parent pair)")
        with pysam.VariantFile(output_vcf_name, "w", header=hdr) as output_vcf:
            for rec in tqdm.tqdm(input_vcf):
                if (rec.chrom, rec.pos) in recalibrated_denovo.index:
                    recs = recalibrated_denovo.loc[[(rec.chrom, rec.pos)]]
                    mq = recs["pair_qual"].min()
                    rec.info["DENOVO_QUAL"] = mq
                output_vcf.write(rec)
    vpu.VcfUtils(None).index_vcf(output_vcf_name)


def main(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Add recalibrated quality (from child/parent calling to the denovo VCF"
    )
    parser.add_argument("denovo_vcf", help="Annotated de novo VCF file", type=str)
    parser.add_argument("recalibrated_vcf", help="Path to the recalibrated VCF file", type=str)

    parser.add_argument(
        "maternal_vcfs", help="JSON file containing dictionary from samples in denovo vcf to maternal vcfs", type=str
    )
    parser.add_argument(
        "paternal_vcfs", help="JSON file containing dictionary from samples in denovo vcf to paternal vcfs", type=str
    )

    args = parser.parse_args(argv)
    with open(args.maternal_vcfs, encoding="utf-8") as f:
        maternal_vcfs = json.load(f)
    with open(args.paternal_vcfs, encoding="utf-8") as f:
        paternal_vcfs = json.load(f)

    parent_df = get_parental_vcf_df(maternal_vcfs, paternal_vcfs)
    recalibrated_denovo = add_parental_qualities_to_denovo_vcf(args.denovo_vcf, parent_df)
    write_recalibrated_vcf(args.denovo_vcf, args.recalibrated_vcf, recalibrated_denovo.sort_index())
