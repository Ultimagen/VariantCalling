#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Collect statistics for a report
# CHANGELOG in reverse chronological order


import argparse
import itertools
import logging
import subprocess

import numpy as np
import pandas as pd
import os

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__ if __name__ != "__main__" else "run_no_gt_report")


def insertion_deletion_statistics(df):
    """
    statistics for indel deletions and insertions.
    We collect for each hmer-indel length the number of insertions and deletions per base (A or G)
    The statistics are collected for homozygous and heterozygous separately
    """
    df_homo = df[(df["gt"] == (1, 1))]
    df_hete = df[(df["gt"] != (1, 1))]

    def calc_hmer_stats(df):
        hmer_stat = {}
        for hlen in range(1, 13):
            df_len = df[df["hmer_indel_length"] == hlen]
            length_col = []
            for ind_clas in ("ins", "del"):
                df_ind = df_len[df_len["indel_classify"] == ind_clas]
                for nuc in (("A", "T"), ("G", "C")):
                    df_nuc = df_ind[(df_ind["hmer_indel_nuc"] == nuc[0]) | (df_ind["hmer_indel_nuc"] == nuc[1])]
                    length_col.append(df_nuc.shape[0])
            hmer_stat[hlen] = length_col
        return pd.DataFrame(data=hmer_stat, index=["ins A", "ins G", "del A", "del G"])

    result = {}
    result["homo"] = calc_hmer_stats(df_homo)
    result["hete"] = calc_hmer_stats(df_hete)
    return result


def allele_freq_hist(df, nbins=100):
    """
    statistics for allele frequency
    For each group return the AF distribution in bins
    """
    bins = np.linspace(0, 1, nbins + 1)
    if "vaf" in df.columns:
        df["af"]=df["vaf"]
    result = {}
    for group in df["variant_type"].unique():
        histogram_data, _ = np.histogram(
            df[df["variant_type"] == group]["af"].apply(lambda x: x[0] if isinstance(x, tuple) else x),
            bins,
        )
        result[group] = pd.Series(histogram_data)
    return pd.DataFrame(data=result)


def snp_statistics(df, ref_fasta):
    import ugvc.vcfbed.variant_annotation as annotation
    from ugvc.dna.utils import revcomp
    # take data from Itai
    df = annotation.classify_indel(df)
    df = annotation.is_hmer_indel(df, ref_fasta)
    df = annotation.get_motif_around(df, 5, ref_fasta)
    df = df.set_index(["chrom", "pos"])

    df["af"] = df["af"].apply(lambda x: x[0])
    df["alt_1"] = df["alleles"].apply(lambda x: x[1])
    df["alt_2"] = df["alleles"].apply(lambda x: x[2] if len(x) > 2 else None)

    def get_by_index(x, index):
        return x[index] if ((x is not None) and (len(x) > index)) else np.nan

    for ind in range(1, 3):
        # pylint: disable=cell-var-from-loop
        df[f"gt_{ind}"] = df["gt"].apply(lambda x: get_by_index(x, ind))

    df = df.drop(columns=["alleles", "gt", "af"])

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

    df_snp["ref_motif"] = df_snp["left_motif"].str.slice(-1) + df_snp["ref"] + df_snp["right_motif"].str.slice(0, 1)

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


def variant_eval_statistics(  # pylint: disable=dangerous-default-value
    vcf_input,
    reference,
    dbsnp,
    output_prefix,
    annotation_names=[],
):
    annotation_names_str = [f"--selectNames {x}" for x in annotation_names]
    annotation_conditions_str = [f'--select vc.hasAttribute("{x}")' for x in annotation_names]
    cmd = (
        [
            "gatk",
            "VariantEval",
            "--eval",
            f"{vcf_input}",
            "--reference",
            f"{reference}",
            "--dbsnp",
            f"{dbsnp}",
            "--output",
            f"{output_prefix}.txt",
        ]
        + annotation_names_str
        + annotation_conditions_str
    )
    logger.info(" ".join(cmd))
    subprocess.check_call(" ".join(cmd).split())

    with open(f"{output_prefix}.txt", "r", encoding="latin-1") as output_file:
        data = _parse_single_report(output_file)
    return data


def _parse_single_report(report):
    def _parse_single_table(input_table):
        headers = input_table.readline().split()
        table = []
        line = input_table.readline().strip("\n").split()
        while len(line) == len(headers):
            table.append(line)
            line = input_table.readline().strip("\n").split()

        return pd.DataFrame(table, columns=headers)

    data = {}
    tables_to_read = pd.Series(
        [
            "CompOverlap",
            "CountVariants",
            "TiTvVariantEvaluator",
            "IndelLengthHistogram",
            "IndelSummary",
            "MetricsCollection",
            "ValidationReport",
            "VariantSummary",
            "MultiallelicSummary",
        ]
    )
    for line in report:
        is_specific_table = tables_to_read.apply(
            # pylint: disable=cell-var-from-loop
            lambda x: x
            in f"#:GATKTable:{x}"
            in line
        )
        start_table = sum(is_specific_table) > 0
        if start_table:
            table_name = tables_to_read[np.where(is_specific_table)[0][0]]
            df = _parse_single_table(report)
            data[table_name] = df
    return data


def run_eval_tables_only(arg_values):
    eval_tables = variant_eval_statistics(
        arg_values.input_file,
        arg_values.reference,
        arg_values.dbsnp,
        arg_values.output_prefix,
        arg_values.annotation_names,
    )
    for eval_table_name, eval_table in eval_tables.items():
        eval_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_{eval_table_name}")


def run_full_analysis(arg_values):
    from ugvc.comparison import vcf_pipeline_utils
    from ugvc.vcfbed import vcftools
    eval_tables = variant_eval_statistics(
        arg_values.input_file,
        arg_values.reference,
        arg_values.dbsnp,
        arg_values.output_prefix,
        [],
    )

    FIELDS_TO_IGNORE = [
        "GNOMAD_AF",
        "X_IC",
        "X_IL",
        "X_HIL",
        "X_HIN",
        "X_LM",
        "X_RM",
        "X_GCC",
        "X_CSS",
        "PL",
        "DB",
        "AD",
    ]
    logger.info("Converting vcf to df")
    df = vcftools.get_vcf_df(
        arg_values.input_file,
        sample_id=arg_values.sample_id,
        sample_name=arg_values.sample_name,
        ignore_fields=FIELDS_TO_IGNORE,
    )
    logger.info("Annotating vcf")
    annotated_df, _ = vcf_pipeline_utils.annotate_concordance(df, arg_values.reference)

    logger.info("insertion/ deletion statistics")
    ins_del_df = insertion_deletion_statistics(df)

    logger.info("Allele frequency histogram")
    af_df = allele_freq_hist(annotated_df)

    logger.info("snps motifs statistics")
    snp_motifs = snp_statistics(df, arg_values.reference)

    logger.info("save all statistics in h5 file")
    ins_del_df["hete"].to_hdf(f"{arg_values.output_prefix}.h5", key="ins_del_hete")
    ins_del_df["homo"].to_hdf(f"{arg_values.output_prefix}.h5", key="ins_del_homo")
    af_df.to_hdf(f"{arg_values.output_prefix}.h5", key="af_hist")
    snp_motifs.to_hdf(f"{arg_values.output_prefix}.h5", key="snp_motifs")
    for eval_table_name, eval_table in eval_tables.items():
        eval_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_{eval_table_name}")



def run_somatic_analysis(arg_values):

    def _get_signatures_from_sig_log(log_file):
        line = log_file.readline()
        while "Composition After Add-Remove" not in line:
            line = log_file.readline()

        signatures = log_file.readline().strip("\n").split()
        # parse the signatures names
        return signatures

    # Install reference
    from SigProfilerMatrixGenerator import install as genInstall
    from SigProfilerAssignment import Analyzer as Analyze
    import sigProfilerPlotting as sigPlt
    from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen
    import shutil

    genInstall.install(arg_values.reference_name)

    # Run SigProfilerAssignment for getting the signatures of the sample
    vcf_dirname = os.path.dirname(arg_values.input_file)

    Analyze.cosmic_fit(samples=vcf_dirname,
                       output=os.path.join(arg_values.output_dir,'sbs'),
                       input_type="vcf",
                       context_type="96",
                       genome_build=arg_values.reference_name,
                       cosmic_version=3.3)

    Analyze.cosmic_fit(samples=vcf_dirname,
                       output=os.path.join(arg_values.output_dir,'id'),
                       input_type="vcf",
                       context_type="ID",
                       genome_build=arg_values.reference_name,
                       collapse_to_SBS96=False,
                       cosmic_version=3.3)

    Analyze.cosmic_fit(samples=vcf_dirname,
                       output=os.path.join(arg_values.output_dir,'dinuc'),
                       input_type="vcf",
                       context_type="DINUC",
                       genome_build=arg_values.reference_name,
                       collapse_to_SBS96=False,
                       cosmic_version=3.3)

    # Plotting the signatures profiles
    sigPlt.plotSBS(
        os.path.join(arg_values.output_dir, 'sbs', 'Assignment_Solution', 'Signatures', 'Assignment_Solution_Signatures.txt'),
        arg_values.output_dir,
        "somatic_sig", "96", percentage=True, savefig_format="png")
    sigPlt.plotID(
        os.path.join(arg_values.output_dir, 'id', 'Assignment_Solution', 'Signatures', 'Assignment_Solution_Signatures.txt'),
        arg_values.output_dir,
        "somatic_sig", "83", percentage=True, savefig_format="png")
    sigPlt.plotDBS(
        os.path.join(arg_values.output_dir, 'dinuc', 'Assignment_Solution', 'Signatures', 'Assignment_Solution_Signatures.txt'),
        arg_values.output_dir,
        "somatic_sig", "78", percentage=True, savefig_format="png")

    # Plotting the sample profile
    matGen.SigProfilerMatrixGeneratorFunc("sample", arg_values.reference_name,
                                          vcf_dirname, plot=True,
                                          exome=False, bed_file=None, chrom_based=False, tsb_stat=False,
                                          seqInfo=False, cushion=100)

    sigPlt.plotSBS(os.path.join(vcf_dirname, "output/SBS/sample.SBS96.all"),
                   arg_values.output_dir,
                   'sample',
                   "96", percentage=False, savefig_format="png")

    sigPlt.plotID(os.path.join(vcf_dirname, "output/ID/sample.ID83.all"),
                  arg_values.output_dir,
                  'sample',
                  "83", percentage=False, savefig_format="png")

    sigPlt.plotDBS(os.path.join(vcf_dirname, "output/DBS/sample.DBS78.all"),
                   arg_values.output_dir,
                   'sample',
                   "78", percentage=False, savefig_format="png")

    # Parse what are the relevant signatures
    with open(os.path.join(arg_values.output_dir, 'sbs', 'Assignment_Solution', 'Solution_Stats', 'Assignment_Solution_Signature_Assignment_log.txt'), "r", encoding="latin-1") as log_file:
        sig_sbs = _get_signatures_from_sig_log(log_file)
        print(sig_sbs)

    with open(os.path.join(arg_values.output_dir, 'id', 'Assignment_Solution', 'Solution_Stats', 'Assignment_Solution_Signature_Assignment_log.txt'), "r", encoding="latin-1") as log_file:
        sig_id = _get_signatures_from_sig_log(log_file)
        print(sig_id)

    with open(os.path.join(arg_values.output_dir, 'dinuc', 'Assignment_Solution', 'Solution_Stats', 'Assignment_Solution_Signature_Assignment_log.txt'), "r", encoding="latin-1") as log_file:
        sig_dinuc = _get_signatures_from_sig_log(log_file)
        print(sig_dinuc)

    final_results_folder = os.path.join(arg_values.output_dir, 'final_results')
    os.mkdir(final_results_folder)

    for sig in sig_sbs:
        shutil.copy(os.path.join(arg_values.output_dir,f'SBS_96_plots_{sig}.png'), final_results_folder)

    for sig in sig_id:
        shutil.copy(os.path.join(arg_values.output_dir,f'ID_83_plots_{sig}.png'), final_results_folder)

    for sig in sig_dinuc:
        shutil.copy(os.path.join(arg_values.output_dir,f'DBS_78_plots_{sig}.png'), final_results_folder)

    vcf_prefix = os.path.basename(arg_values.input_file).split('.')[0]
    shutil.copy(os.path.join(arg_values.output_dir, f'SBS_96_plots_{vcf_prefix}.png'), final_results_folder)
    shutil.copy(os.path.join(arg_values.output_dir, f'ID_83_plots_{vcf_prefix}.png'), final_results_folder)
    shutil.copy(os.path.join(arg_values.output_dir, f'DBS_78_plots_{vcf_prefix}.png'), final_results_folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="run_no_gt_report.py",
        description="Collect metrics for runs without ground truth",
    )

    subparsers = ap.add_subparsers()
    full_analysis = subparsers.add_parser(name="full_analysis", description="Run the full analysis of no_gt_report")

    full_analysis.add_argument("--input_file", help="Input vcf file", required=True, type=str)
    full_analysis.add_argument("--dbsnp", help="dbsnp vcf file", required=True, type=str)
    full_analysis.add_argument("--reference", help="Reference genome", required=True, type=str)
    full_analysis.add_argument("--output_prefix", help="output file", required=True, type=str)
    full_analysis.add_argument(
        "--sample_id",
        help="sample id (useful in case there is more than one in the vcf)",
        required=False,
        type=int,
        default=0,
    )
    full_analysis.add_argument(
        "--sample_name", help="sample name (useful in case there is more than one in the vcf)", required=False, type=str
    )
    full_analysis.set_defaults(func=run_full_analysis)

    # Run variant eval only - for JC report
    parser_eval_tables = subparsers.add_parser(name="variant_eval", description="Run variant eval only")
    parser_eval_tables.add_argument("--input_file", help="Input vcf file", required=True, type=str)
    parser_eval_tables.add_argument("--dbsnp", help="dbsnp vcf file", required=True, type=str)
    parser_eval_tables.add_argument("--reference", help="Reference genome", required=True, type=str)
    parser_eval_tables.add_argument("--output_prefix", help="output file", required=True, type=str)
    parser_eval_tables.add_argument(
        "--annotation_names",
        help="annotation name to filter on",
        required=False,
        nargs="*",
    )
    parser_eval_tables.set_defaults(func=run_eval_tables_only)


    # Run somatic analysis on somatic data
    full_analysis = subparsers.add_parser(name="somatic_analysis", description="Run mutation signatures and motif graphs")

    full_analysis.add_argument("--input_file", help="Input vcf file", required=True, type=str)
    full_analysis.add_argument("--reference_name", help="Reference genome name", required=False, type=str, default="GRCh38")
    full_analysis.add_argument("--output_dir", help="output directory", required=True, type=str)
    full_analysis.set_defaults(func=run_somatic_analysis)


    args = ap.parse_args()
    args.func(args)
