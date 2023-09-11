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

import ugvc.vcfbed.variant_annotation as annotation
from ugvc.comparison import vcf_pipeline_utils
from ugvc.dna.utils import revcomp
from ugvc.vcfbed import vcftools

from SigProfilerMatrixGenerator import install as genInstall
from SigProfilerAssignment import Analyzer as Analyze
import sigProfilerPlotting as sigPlt
from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as matGen
import shutil

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


def bcftools_stats_statistics(
        vcf_input,
        output_prefix
):
    # novel only vcf
    cmd = f"bcftools view -n {vcf_input} | bcftools stats --af-bins 0,0.25,0.75,1 --af-tag AF > {output_prefix}_novel.txt"

    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    cmd = f"bcftools view -k {vcf_input} | bcftools stats --af-bins 0,0.25,0.75,1 --af-tag AF > {output_prefix}_known.txt"

    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    data_novel,_ ,_,df_novel_idd = _parse_stats_report(f"{output_prefix}_novel.txt")

    data_known,_ ,_,df_known_idd = _parse_stats_report(f"{output_prefix}_known.txt")
    data_novel.loc['indel_novelty_rate', 'values'] = 100
    data_known.loc['indel_novelty_rate', 'values'] = 0

    data_all = data_novel+data_known
    data_all.loc['SNP_to_indel_ratio', 'values'] = data_all.loc['nSnps', 'values'] / data_all.loc['nIndels', 'values']
    data_all.loc['tiTvRatio','values'] = data_all.loc['nTi', 'values'] / data_all.loc['nTv', 'values']
    data_all.loc['insertion_to_deletion_ratio', 'values'] = data_all.loc['nInsertions', 'values'] / data_all.loc['nDeletions', 'values']
    data_all.loc['indel_novelty_rate', 'values'] = data_novel.loc['nIndels', 'values']/(data_novel.loc['nIndels', 'values'] + data_known.loc['nIndels', 'values'])*100
    # group them together
    data = {}
    data['novel'] = data_novel
    data['known'] = data_known
    data['all'] = data_all
    return data, df_novel_idd, df_known_idd



def _parse_stats_report(report):
    df_sn = _parse_stats_section_report(report, "SN")
    df_tstv = _parse_stats_section_report(report, "TSTV")
    df_idd = _parse_stats_section_report(report, "IDD")
    df_af = _parse_stats_section_report(report, "AF")
    # arrange the relevant results in one dataframe
    n_records = int(df_sn.loc[df_sn['[3]key']=='number of records:','[4]value'])
    n_snps = int(df_sn.loc[df_sn['[3]key'] == 'number of SNPs:', '[4]value'])
    n_indels = int(df_sn.loc[df_sn['[3]key'] == 'number of indels:', '[4]value'])
    nti = int(df_tstv['[3]ts'])
    ntv = int(df_tstv['[4]tv'])
    ti_tv_ratio = float(df_tstv['[5]ts/tv'])
    multiallelic_snps = int(df_sn.loc[df_sn['[3]key'] == 'number of multiallelic SNP sites:', '[4]value'])
    multiallelic = int(df_sn.loc[df_sn['[3]key'] == 'number of multiallelic sites:', '[4]value'])
    try:
        hom_vars = int(df_af.loc[df_af['[3]allele frequency']=='0.875000','[4]number of SNPs']) + int(df_af.loc[df_af['[3]allele frequency']=='0.875000','[7]number of indels'])
    except(IndexError, TypeError):
        hom_vars = 0
    n_deletions = np.sum(df_idd.loc[(df_idd['[3]length (deletions negative)']).astype(int) < 0,'[4]number of sites'].astype(int))
    n_insertions = np.sum(df_idd.loc[(df_idd['[3]length (deletions negative)']).astype(int) > 0, '[4]number of sites'].astype(int))

    index_values = ['nRecords','nDeletions','nInsertions','nSnps','nIndels','nTi','nTv','tiTvRatio',
             'SNP_to_indel_ratio','indel_novelty_rate','insertion_to_deletion_ratio','nHets','nHomVar',
             'nMultiSNPs','nMultiIndels','nBiallelicSNPs','nBiallelicIndels'
             ]
    data = [n_records,
            n_deletions,
            n_insertions,
            n_snps,
            n_indels,
            nti,
            ntv,
            ti_tv_ratio,
            n_snps/n_indels,
            0,
            n_insertions/n_deletions,
            n_records - hom_vars,
            hom_vars,
            multiallelic_snps,
            multiallelic - multiallelic_snps,
            n_snps - multiallelic_snps,
            n_indels - (multiallelic - multiallelic_snps)
            ]

    df = pd.DataFrame(data, columns=["values"], index=index_values)
    return df, df_sn, df_tstv, df_idd


def _parse_stats_section_report(filename, section):
    # Read file into memory
    with open(filename, 'r') as file:
        line = file.readline()
        while f"# {section}\t" not in line:
            line = file.readline()
        headers = (line.strip().split('\t'))
        sec_table = []
        for line in file:
            if line.startswith(f"{section}"):
                sec_table.append(line.strip().split('\t'))
    df = pd.DataFrame(sec_table, columns=headers)
    return df



def insertion_deletion_statistics(vcf_input, output_prefix):
    # extract info
    # heterozygous
    cmd = f"bcftools query -i \'(INFO/VARIANT_TYPE=\"h-indel\" || INFO/VARIANT_TYPE=\"non-h-indel\") && GT=\"0/1\"\' -f \'%INFO/X_HIL;%INFO/X_HIN;%INFO/X_IC\\n\' {vcf_input} > {output_prefix}_info_hete.txt"
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    # homozygous
    cmd = f"bcftools query -i \'(INFO/VARIANT_TYPE=\"h-indel\" || INFO/VARIANT_TYPE=\"non-h-indel\") && GT=\"1/1\"\' -f \'%INFO/X_HIL;%INFO/X_HIN;%INFO/X_IC\\n\' {vcf_input} > {output_prefix}_info_homo.txt"
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    def _create_ins_del_summary_table(filename):
        logger.info("Reading csv to df")
        filename = filename
        column_names = ['X_HIL', 'X_HIN', 'X_IC']
        df = pd.read_csv(filename, delimiter=";", names=column_names)

        logger.info("Creating a summary table of insertion and deletions by base")
        # remove multi-allelic
        df = df.loc[df["X_HIL"].str.split(',').apply(lambda x: (len(x) == 1))]
        # make it as a summary table
        summary = df.groupby(['X_HIN', 'X_HIL', 'X_IC']).size().reset_index(name='Counts')
        summary.loc[summary['X_HIN'] == 'T', 'X_HIN'] = 'A'
        summary.loc[summary['X_HIN'] == 'C', 'X_HIN'] = 'G'
        summary['index_name'] = summary['X_IC'] + ' ' + summary['X_HIN']
        summary = summary.groupby(['index_name', 'X_HIL'])['Counts'].sum().reset_index(name='Counts')
        summary_table = summary.pivot(index='index_name', columns='X_HIL', values='Counts').fillna(0)
        summary_table = summary_table.astype(int)
        num_ins_del = summary_table.sum().sum()
        valid_columns = set(summary_table.columns) & set(pd.Series(np.arange(1, 13)).apply(str))
        summary_table.drop(['del .', 'ins .'], errors='ignore', inplace=True)
        summary_table = summary_table[sorted(list(valid_columns), key=int)]

        return summary_table, num_ins_del

    summary_table_hete, num_ins_del_hete = _create_ins_del_summary_table(f"{output_prefix}_info_hete.txt")
    summary_table_homo, num_ins_del_homo = _create_ins_del_summary_table(f"{output_prefix}_info_homo.txt")
    print(num_ins_del_hete)
    print(num_ins_del_homo)
    indel_het_to_hom_ratio =num_ins_del_hete/num_ins_del_homo

    return summary_table_hete, summary_table_homo, indel_het_to_hom_ratio




def run_full_analysis(arg_values):
    # PASS only
    cmd = f"bcftools view -f 'PASS,.' {arg_values.input_file} -Oz -o {arg_values.output_prefix}_PASS.vcf.gz"
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    stats_tables, df_novel_idd, df_known_idd = bcftools_stats_statistics(f"{arg_values.output_prefix}_PASS.vcf.gz",arg_values.output_prefix)

    # insertion and deleletion by base
    ins_del_hete, ins_del_homo, indel_het_to_hom_ratio = insertion_deletion_statistics(f"{arg_values.output_prefix}_PASS.vcf.gz", arg_values.output_prefix)

    # same for exome
    cmd = f"bcftools view -i \'INFO/EXOME=\"TRUE\"\' {arg_values.output_prefix}_PASS.vcf.gz -Oz -o {arg_values.output_prefix}_exome.vcf.gz"
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    stats_tables_exome, df_novel_idd_exome, df_known_idd_exome = bcftools_stats_statistics(f"{arg_values.output_prefix}_exome.vcf.gz",
                                                                         f"{arg_values.output_prefix}_exome")



    # same for ug_hcr
    cmd = f"bcftools view -i \'INFO/UG_HCR=\"TRUE\"\' {arg_values.output_prefix}_PASS.vcf.gz -Oz -o {arg_values.output_prefix}_ug_hcr.vcf.gz"
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    stats_tables_ug_hcr, df_novel_idd_ug_hcr, df_known_idd_ug_hcr = bcftools_stats_statistics(f"{arg_values.output_prefix}_ug_hcr.vcf.gz",
                                                                         f"{arg_values.output_prefix}_ug_hcr")


    logger.info("save all statistics in h5 file")
    for stats_table_name, stats_table in stats_tables.items():
        stats_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_{stats_table_name}")

    df_novel_idd.to_hdf(f"{arg_values.output_prefix}.h5", key="idd_novel")
    df_known_idd.to_hdf(f"{arg_values.output_prefix}.h5", key="idd_known")

    ins_del_hete.to_hdf(f"{arg_values.output_prefix}.h5", key="ins_del_hete")
    ins_del_homo.to_hdf(f"{arg_values.output_prefix}.h5", key="ins_del_homo")
    # exome
    for stats_table_name, stats_table in stats_tables_exome.items():
        stats_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_exome_{stats_table_name}")

    # hg_hcr
    for stats_table_name, stats_table in stats_tables_ug_hcr.items():
        stats_table.to_hdf(f"{arg_values.output_prefix}.h5", key=f"eval_hg_hcr_{stats_table_name}")


    # F1 prediction
    # number of LowQualInExome
    cmd = f"bcftools view -H {arg_values.input_file} | awk \'{{print $7}}\' | sort | uniq -c | grep LowQualInExome| awk \'{{print $1}}\'"
    logger.info(cmd)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    number_of_LowQualInExome = int(output.strip())
    logger.info(f"number of LowQualInExome:{number_of_LowQualInExome}")

    # indel SOR > 3 proportion
    cmd = f"bcftools view {arg_values.output_prefix}_PASS.vcf.gz -H -v indels | awk \'{{print $8}}\' | sed \'s/;/\t/g\' | sed \'s/.*SOR=/ /\' | awk \'$1>3{{s=s+1}} END{{print s/NR}}\'"
    logger.info(cmd)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    indels_SOR_above_3_proportion = float(output.strip())
    logger.info(f"indel SOR > 3 proportion':{indels_SOR_above_3_proportion}")

    # quantiles:20_percentile_RS
    cmd = f"zcat {arg_values.output_prefix}_PASS.vcf.gz | grep -v \'^#\' | awk \'{{print $6}}\' | sort -g | perl -ne \'$arr[$. - 1] = $_; END {{ print $arr[int($./5)] }}\'"
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    the20_percentile_RS = float(output.strip())
    logger.info(f"20th percentile RS':{the20_percentile_RS}")

    # InsertionDeletionRatio:UG_HCR:known
    insertion_to_deletion_ratio = stats_tables_ug_hcr["known"].loc["insertion_to_deletion_ratio"][0]
    logger.info(f"Insertion deletion ratio in UG_HCR known':{the20_percentile_RS}")

    # IndelSummary:novel:Average of indel_het_to_hom_ratio
    logger.info(f"Indel heterozygous homozygous ratio':{indel_het_to_hom_ratio}")


def run_somatic_analysis(arg_values):

    def _get_signatures_from_sig_log(log_file):
        line = log_file.readline()
        while "Composition After Add-Remove" not in line:
            line = log_file.readline()

        signatures = log_file.readline().strip("\n").split()
        activity_values = log_file.readline().strip("\n").split()[1:]
        # parse the signatures names
        return signatures, activity_values

    # Install reference
    genInstall.install(arg_values.reference_name)

    # Run SigProfilerAssignment for getting the signatures of the sample
    vcf_dirname = os.path.dirname(arg_values.input_file)
    output_dir = arg_values.output_prefix + '/'
    Analyze.cosmic_fit(samples=vcf_dirname,
                       output=os.path.join(output_dir,'sbs'),
                       input_type="vcf",
                       context_type="96",
                       genome_build=arg_values.reference_name,
                       cosmic_version=arg_values.cosmic_version)

    Analyze.cosmic_fit(samples=vcf_dirname,
                       output=os.path.join(output_dir,'id'),
                       input_type="vcf",
                       context_type="ID",
                       genome_build=arg_values.reference_name,
                       collapse_to_SBS96=False,
                       cosmic_version=arg_values.cosmic_version)

    Analyze.cosmic_fit(samples=vcf_dirname,
                       output=os.path.join(output_dir,'dinuc'),
                       input_type="vcf",
                       context_type="DINUC",
                       genome_build=arg_values.reference_name,
                       collapse_to_SBS96=False,
                       cosmic_version=arg_values.cosmic_version)

    # Plotting the signatures profiles
    sigPlt.plotSBS(
        os.path.join(output_dir, 'sbs', 'Assignment_Solution', 'Signatures', 'Assignment_Solution_Signatures.txt'),
        output_dir,
        "somatic_sig", "96", percentage=True, savefig_format="png")
    sigPlt.plotID(
        os.path.join(output_dir, 'id', 'Assignment_Solution', 'Signatures', 'Assignment_Solution_Signatures.txt'),
        output_dir,
        "somatic_sig", "83", percentage=True, savefig_format="png")
    sigPlt.plotDBS(
        os.path.join(output_dir, 'dinuc', 'Assignment_Solution', 'Signatures', 'Assignment_Solution_Signatures.txt'),
        output_dir,
        "somatic_sig", "78", percentage=True, savefig_format="png")

    # Plotting the sample profile
    matGen.SigProfilerMatrixGeneratorFunc("sample", arg_values.reference_name,
                                          vcf_dirname, plot=True,
                                          exome=False, bed_file=None, chrom_based=False, tsb_stat=False,
                                          seqInfo=False, cushion=100)

    sigPlt.plotSBS(os.path.join(vcf_dirname, "output/SBS/sample.SBS96.all"),
                   output_dir,
                   'sample',
                   "96", percentage=False, savefig_format="png")

    sigPlt.plotID(os.path.join(vcf_dirname, "output/ID/sample.ID83.all"),
                  output_dir,
                  'sample',
                  "83", percentage=False, savefig_format="png")

    sigPlt.plotDBS(os.path.join(vcf_dirname, "output/DBS/sample.DBS78.all"),
                   output_dir,
                   'sample',
                   "78", percentage=False, savefig_format="png")

    # Parse what are the relevant signatures
    with open(os.path.join(output_dir, 'sbs', 'Assignment_Solution', 'Solution_Stats', 'Assignment_Solution_Signature_Assignment_log.txt'), "r", encoding="latin-1") as log_file:
        sig_sbs,activity_values_sbs = _get_signatures_from_sig_log(log_file)
        print(sig_sbs)

    with open(os.path.join(output_dir, 'id', 'Assignment_Solution', 'Solution_Stats', 'Assignment_Solution_Signature_Assignment_log.txt'), "r", encoding="latin-1") as log_file:
        sig_id,activity_values_id = _get_signatures_from_sig_log(log_file)
        print(sig_id)

    with open(os.path.join(output_dir, 'dinuc', 'Assignment_Solution', 'Solution_Stats', 'Assignment_Solution_Signature_Assignment_log.txt'), "r", encoding="latin-1") as log_file:
        sig_dinuc,activity_values_dinuc = _get_signatures_from_sig_log(log_file)
        print(sig_dinuc)

    final_results_folder = os.path.join(output_dir, 'final_results')
    os.mkdir(final_results_folder)

    # copy the signatures pngs
    for sig in sig_sbs:
        shutil.copy(os.path.join(output_dir,f'SBS_96_plots_{sig}.png'), final_results_folder)

    for sig in sig_id:
        shutil.copy(os.path.join(output_dir,f'ID_83_plots_{sig}.png'), final_results_folder)

    for sig in sig_dinuc:
        shutil.copy(os.path.join(output_dir,f'DBS_78_plots_{sig}.png'), final_results_folder)

    # create signatures activity files
    activities_sbs = pd.Series(activity_values_sbs, index=sig_sbs)
    activities_sbs.to_hdf(f"{arg_values.output_prefix}.h5", key="SBS96_activity")

    activities_id = pd.Series(activity_values_id, index=sig_id)
    activities_id.to_hdf(f"{arg_values.output_prefix}.h5", key="ID83_activity")

    activities_dinuc = pd.Series(activity_values_dinuc, index=sig_dinuc)
    activities_dinuc.to_hdf(f"{arg_values.output_prefix}.h5", key="DBS78_activity")

    # copy the sample profile pngs
    vcf_prefix = os.path.basename(arg_values.input_file).split('.')[0]
    shutil.copy(os.path.join(output_dir, f'SBS_96_plots_{vcf_prefix}.png'), final_results_folder)
    shutil.copy(os.path.join(output_dir, f'ID_83_plots_{vcf_prefix}.png'), final_results_folder)
    shutil.copy(os.path.join(output_dir, f'DBS_78_plots_{vcf_prefix}.png'), final_results_folder)

    # save the sample profile to h5
    sig_profile_sbs = pd.read_csv(os.path.join(vcf_dirname, "output/SBS/sample.SBS96.all"), sep='\t')
    sig_profile_sbs.to_hdf(f"{arg_values.output_prefix}.h5", key="SBS96_profile")

    sig_profile_id = pd.read_csv(os.path.join(vcf_dirname, "output/ID/sample.ID83.all"), sep='\t')
    sig_profile_id.to_hdf(f"{arg_values.output_prefix}.h5", key="ID83_profile")

    sig_profile_dbs = pd.read_csv(os.path.join(vcf_dirname, "output/DBS/sample.DBS78.all"), sep='\t')
    sig_profile_dbs.to_hdf(f"{arg_values.output_prefix}.h5", key="DBS78_profile")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="run_no_gt_report.py",
        description="Collect metrics for runs without ground truth",
    )

    subparsers = ap.add_subparsers()
    full_analysis = subparsers.add_parser(name="full_analysis", description="Run the full analysis of no_gt_report")

    full_analysis.add_argument("--input_file", help="Input vcf file", required=True, type=str)
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
    somatic_analysis = subparsers.add_parser(name="somatic_analysis", description="Run mutation signatures and motif graphs")

    somatic_analysis.add_argument("--input_file", help="Input unzipped vcf file", required=True, type=str)
    somatic_analysis.add_argument("--reference_name", help="Reference genome name", required=False, type=str, default="GRCh38")
    somatic_analysis.add_argument("--output_prefix", help="output file and directory", required=True, type=str)
    somatic_analysis.add_argument("--cosmic_version", help="Signatures cosmic version", required=False, type=str, default=3.3)
    somatic_analysis.set_defaults(func=run_somatic_analysis)


    args = ap.parse_args()
    args.func(args)
