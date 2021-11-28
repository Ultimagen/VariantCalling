import pathmagic
import argparse
import logging
import pandas as pd
import subprocess
import python.vcftools as vcftools
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import python.modules.variant_annotation as annotation
import numpy as np
import itertools
from python.utils import revcomp

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__ if __name__ != "__main__" else "run_no_gt_report")

def insertion_deletion_statistics(df):
    '''
    statistics for indel deletions and insertions.
    We collect for each hmer-indel length the number of insertions and deletions per base (A or G)
    The statistics are collected for homozygous and heterozygous separately
    '''
    df_homo = df[(df['gt'] == (1, 1))]
    df_hete = df[(df['gt'] != (1, 1))]

    def calc_hmer_stats(df):
        hmer_stat = {}
        for hlen in range(1,13):
            df_len = df[df['hmer_indel_length']==hlen]
            length_col = []
            for ind_clas in ['ins','del']:
                df_ind = df_len[df_len['indel_classify']==ind_clas]
                for nuc in [('A','T'),('G','C')]:
                    df_nuc = df_ind[(df_ind['hmer_indel_nuc'] == nuc[0]) | (df_ind['hmer_indel_nuc'] == nuc[1])]
                    length_col.append(df_nuc.shape[0])
            hmer_stat[hlen] = length_col
        return pd.DataFrame(data=hmer_stat,index=['ins A','ins G','del A','del G'])

    result = {}
    result['homo']=calc_hmer_stats(df_homo)
    result['hete']=calc_hmer_stats(df_hete)
    return result

def allele_freq_hist(df, nbins = 100):
    '''
    statistics for allele frequency
    For each group return the AF distribution in bins
    '''
    bins = np.linspace(0, 1, nbins+1)
    result = {}
    for group in df['variant_type'].unique():
        histogram_data, _ = np.histogram(df[df['variant_type'] == group]['af'].apply(lambda x: x[0] if type(x) == tuple else x), bins)
        result[group] = pd.Series(histogram_data)
    return pd.DataFrame(data=result)

def snp_statistics(df, ref_fasta):
    ## take data from Itai
    df = annotation.classify_indel(df)
    df = annotation.is_hmer_indel(df, ref_fasta)
    df = annotation.get_motif_around(df, 5, ref_fasta)
    df = df.set_index(["chrom", "pos"])

    df["af"] = df["af"].apply(lambda x: x[0])
    df["alt_1"] = df["alleles"].apply(lambda x: x[1])
    df["alt_2"] = df["alleles"].apply(lambda x: x[2] if len(x) > 2 else None)

    for j in range(1, 7):
        df[f"pl_{j}"] = df["pl"].apply(lambda x: x[j] if ((x is not None) and (len(x) > j)) else np.nan)

    for j in range(1, 3):
        df[f"gt_{j}"] = df["gt"].apply(lambda x: x[j] if len(x) > j else np.nan)

    for j in range(1, 4):
        df[f"ad_{j}"] = df["ad"].apply(lambda x: x[j] if len(x) > j else np.nan)

    df = df.drop(columns=["alleles", "gt", "pl", "ad", "af"])

    motif_index_bothstrands = pd.MultiIndex.from_tuples(
        [
            x
            for x in itertools.product(
            ["".join(x) for x in itertools.product(["A", "C", "G", "T"], repeat=3)],
            ["A", "C", "G", "T"],
        )
            if x[0][1] != x[1]
        ],
        names=["ref_motif", "alt_1"],
    )
    motif_index = pd.MultiIndex.from_tuples(
        [
            x
            for x in itertools.product(
            ["".join(x) for x in itertools.product(["A", "C", "G", "T"], repeat=3)],
            ["A", "C", "G", "T"],
        )
            if x[0][1] != x[1] and (x[0][1] == "A" or x[0][1] == "C")
        ],
        names=["ref_motif", "alt_1"],
    )

    df_snp = df[~df["indel"]].drop(
        columns=[
            "indel",
            "indel_classify",
            "indel_length",
            "hmer_indel_length",
            "hmer_indel_nuc",
        ]
    )

    df_snp["ref_motif"] = (
            df_snp["left_motif"].str.slice(-1)
            + df_snp["ref"]
            + df_snp["right_motif"].str.slice(0, 1)
    )

    motifs_bothstrands = (
        df_snp.groupby(["ref_motif", "alt_1"])
            .size()
            .reindex(motif_index_bothstrands)
            .fillna(0)
            .astype(int)
            .rename("size")
    )

    index_vars = motifs_bothstrands.index.names
    x = motifs_bothstrands.reset_index()
    x_f = (
        x[x["ref_motif"].str.slice(1, 2).isin(["A", "C"])]
            .set_index(index_vars)
            .reindex(motif_index)
            .fillna(0)
            .astype(int)
    )
    x_r = x[x["ref_motif"].str.slice(1, 2).isin(["G", "T"])]
    x_r = (
        x_r.assign(**{c: x_r[c].apply(revcomp) for c in index_vars})
            .set_index(index_vars)
            .reindex(motif_index)
            .fillna(0)
            .astype(int)
    )
    motifs = (x_f + x_r)["size"]
    return motifs

def variant_eval_statistics(vcf_input, reference, dbsnp, output_prefix):
    cmd = ["gatk", f"VariantEval",
           "--eval", f"{vcf_input}",
           "--reference", f"{reference}",
           "--dbsnp", f"{dbsnp}",
           "--output", f"{output_prefix}.txt"]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

    with open(f"{output_prefix}.txt") as f:
        data = _parse_single_report(f)
    return data

def _parse_single_report(f):
    def _parse_single_table(f):
        headers = f.readline().split()
        table = []
        l = f.readline().strip("\n").split()
        while len(l) == len(headers):
            table.append(l)
            l = f.readline().strip("\n").split()

        return pd.DataFrame(table, columns=headers)

    data = dict()
    tables_to_read = pd.Series(["CompOverlap",
                                "CountVariants",
                                "TiTvVariantEvaluator",
                                "IndelLengthHistogram",
                                "IndelSummary",
                                "MetricsCollection",
                                "ValidationReport",
                                "VariantSummary",
                                "MultiallelicSummary"])
    for l in f:
        is_specific_table = tables_to_read.apply(lambda x: x in f"#:GATKTable:{x}" in l)
        start_table = sum(is_specific_table) > 0
        if start_table:
            table_name = tables_to_read[np.where(is_specific_table)[0][0]]
            df = _parse_single_table(f)
            data[table_name] = df
    return data

def run_eval_tables_only(args):
    eval_tables = variant_eval_statistics(args.input_file, args.reference, args.dbsnp, args.output_prefix)
    for eval_table_name in eval_tables.keys():
        eval_tables[eval_table_name].to_hdf(f"{args.output_prefix}.h5", key=f"eval_{eval_table_name}")

def run_full_analysis(args):
    eval_tables = variant_eval_statistics(args.input_file, args.reference, args.dbsnp, args.output_prefix)

    logger.info("Converting vcf to df")
    df = vcftools.get_vcf_df(args.input_file)
    logger.info("Annotating vcf")
    annotated_df, _ = vcf_pipeline_utils.annotate_concordance(
        df, args.reference)

    logger.info("insertion/ deletion statistics")
    ins_del_df = insertion_deletion_statistics(df)

    logger.info("Allele frequency histogram")
    af_df = allele_freq_hist(annotated_df)

    logger.info("snps motifs statistics")
    snp_motifs = snp_statistics(df, args.reference)

    logger.info("save all statistics in h5 file")
    ins_del_df['hete'].to_hdf(f"{args.output_prefix}.h5", key="ins_del_hete")
    ins_del_df['homo'].to_hdf(f"{args.output_prefix}.h5", key="ins_del_homo")
    af_df.to_hdf(f"{args.output_prefix}.h5", key="af_hist")
    snp_motifs.to_hdf(f"{args.output_prefix}.h5", key="snp_motifs")
    for eval_table_name in eval_tables.keys():
        eval_tables[eval_table_name].to_hdf(f"{args.output_prefix}.h5", key=f"eval_{eval_table_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="run_no_gt_report.py", description="Collect metrics for runs without ground truth")

    subparsers = ap.add_subparsers()
    full_analysis = subparsers.add_parser(name='full_analysis',
                                               description='Run teh full analysis of no_gt_report')

    full_analysis.add_argument("--input_file", help="Input vcf file",
                    required=True, type=str)
    full_analysis.add_argument("--dbsnp", help="dbsnp vcf file",
                    required=True, type=str)
    full_analysis.add_argument("--reference", help='Reference genome',
                    required=True, type=str)
    full_analysis.add_argument("--output_prefix", help="output file",
                    required=True, type=str)
    full_analysis.set_defaults(func=run_full_analysis)

    # Run variant eval only - for JC repor
    parser_eval_tables = subparsers.add_parser(name='variant_eval',
                                               description='Run variant eval only')
    parser_eval_tables.add_argument("--input_file", help="Input vcf file",
                    required=True, type=str)
    parser_eval_tables.add_argument("--dbsnp", help="dbsnp vcf file",
                    required=True, type=str)
    parser_eval_tables.add_argument("--reference", help='Reference genome',
                    required=True, type=str)
    parser_eval_tables.add_argument("--output_prefix", help="output file",
                    required=True, type=str)
    parser_eval_tables.set_defaults(func=run_eval_tables_only)

    args = ap.parse_args()
    args.func(args)




