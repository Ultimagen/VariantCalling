import pandas as pd
from os.path import join as pjoin
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
from ugvc.vcfbed import variant_annotation
import warnings
warnings.filterwarnings('ignore')


def load_classify_gt_counts(concord_file, candidate_folder, original_lcr, chrom, exome):
    """
    cladiffy variants by lcr per chromosome per sample
    Parameters
    ----------
    concord_file - concordance file
    candidate_folder - lcr candidate folder with ug_lcr_v2.bed file in it.
    original_lcr - (boolean) weather to use original lcr or not.
    chrom - chromosome
    exome - (boolean) wetaher to clasiffy variants only on exome or not.

    Returns
    -------
    dataframe with tp,fp,fn classifications.
    """
    # t0 = time.time()
    concord_df = pd.read_hdf(concord_file, key=chrom)

    # print("read_hdf " + chrom + ": "+ str(time.time() - t0))

    concord_df = concord_df.rename(columns={"exome.twist": "exome_twist"})
    if original_lcr:
        concord_df, an = variant_annotation.annotate_intervals(concord_df, "/data/UG_HCR_recalc/ug_hcr_v1.4/ug_lcr.bed")
    else:
        concord_df, an = variant_annotation.annotate_intervals(concord_df,
                                                               os.path.join(candidate_folder, "ug_lcr_v2.bed"))
    if exome:
        concord_df = concord_df.query("exome_twist")

    # look at unfiltered variants only
    # concord_df = concord_df.query("filter=='PASS'")

    if original_lcr:
        qstr = "~ug_lcr"
    else:
        qstr = "~ug_lcr_v2"

    if qstr != '':
        concord_df = concord_df.query(qstr)

    # t = time.time()
    count_df = concord_df.groupby(["indel", "classify_gt"])["chrom"] \
        .agg("count") \
        .reset_index() \
        .rename(columns={"chrom": "count"}) \
        .pivot(index='indel', columns='classify_gt')['count'] \
        .reset_index() \
        .assign(chrom=chrom)
    # print("groupby_agg " + chrom + ": "+ str(time.time() - t))
    # print("load_classify_gt_counts " + chrom + ": "+ str(time.time() - t0))
    return count_df

def stats_over_all_chroms(concord_file, candidate_folder, original_lcr, exome):
    """
    gather clasiffting stats over all chromosome per sample
    Parameters
    ----------
    concord_file - concordance file
    candidate_folder - lcr candidate folder with ug_lcr_v2.bed file in it.
    original_lcr - (boolean) weather to use original lcr or not.
    exome - (boolean) wetaher to clasiffy variants only on exome or not.

    Returns
    -------
    datafreme with stats(precision,recall,F1) per SNP/Indel per WGS/exome per original/new lcr.
    """
    print(f"{original_lcr}, {exome}")
    count_df = pd.concat(
        [load_classify_gt_counts(concord_file, candidate_folder, original_lcr, "chr" + str(c), exome) for c in
         range(1, 23)])
    agg_df = count_df.groupby("indel") \
        .agg("sum") \
        .assign(recall=lambda df: df["tp"] / (df["tp"] + df["fn"]),
                precision=lambda df: df["tp"] / (df["tp"] + df["fp"]),
                F1=lambda df: 2 * (df["recall"] * df["precision"]) / (df["recall"] + df["precision"])) \
        .reset_index() \
        .assign(type=lambda df: ["Indel" if ind else "SNP" for ind in df["indel"]]) \
        .drop(["indel"], axis=1) \
        .assign(original_lcr=original_lcr, exome=exome)

    return agg_df

def plot_stats_summary(df_all_samples_stats):
    """
    plot performance statistics summary over all given samples
    Parameters
    ----------
    df_all_samples_stats - dataframe with stats(precision,recall,F1) per SNP/Indel per WGS/exome per original/new lcr per sample.

    Returns
    -------
    outputs a summary plot
    """
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2)

    # width of the bars
    barWidth = 0.3

    ######## WGS SNP ########
    # original lcr : WGS,SNP
    bars1 = df_all_samples_stats[(df_all_samples_stats['exome']==False) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='SNP')]['F1'].values
    # new lcr : WGS,SNP
    bars2 = df_all_samples_stats[(df_all_samples_stats['exome']==False) \
                                 & (df_all_samples_stats['original_lcr']==False) \
                                 & (df_all_samples_stats['type']=='SNP')]['F1'].values
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Create blue bars
    ax1.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='v1.4')
    # Create cyan bars
    ax1.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',  capsize=7, label='lcr_can1')

    # general layout
    sample_names=df_all_samples_stats[(df_all_samples_stats['exome']==False) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='SNP')]['gt_sample_name'].values

    plt.sca(ax1)
    plt.xticks([r + barWidth for r in range(len(bars1))], sample_names,rotation=90)
    plt.ylabel('SNP F1')
    plt.xlabel('sample name')
    plt.ylim([0.99,1])
    plt.title('WGS')

    ######## WGS Indel ########
    bars1 = df_all_samples_stats[(df_all_samples_stats['exome']==False) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='Indel')]['F1'].values
    # new lcr : WGS,SNP
    bars2 = df_all_samples_stats[(df_all_samples_stats['exome']==False) \
                                 & (df_all_samples_stats['original_lcr']==False) \
                                 & (df_all_samples_stats['type']=='Indel')]['F1'].values

    ax3.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='v1.4')
    # Create cyan bars
    ax3.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',  capsize=7, label='lcr_can1')
    sample_names=df_all_samples_stats[(df_all_samples_stats['exome']==False) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='Indel')]['gt_sample_name'].values
    plt.sca(ax3)
    plt.xticks([r + barWidth for r in range(len(bars1))], sample_names,rotation=90)
    plt.ylabel('Indel F1')
    plt.xlabel('sample name')
    plt.ylim([0.98,0.995])
    plt.title('WGS')

    ######## Exome SNP ########
    bars1 = df_all_samples_stats[(df_all_samples_stats['exome']==True) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='SNP')]['F1'].values
    # new lcr : WGS,SNP
    bars2 = df_all_samples_stats[(df_all_samples_stats['exome']==True) \
                                 & (df_all_samples_stats['original_lcr']==False) \
                                 & (df_all_samples_stats['type']=='SNP')]['F1'].values

    ax2.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='v1.4')
    # Create cyan bars
    ax2.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',  capsize=7, label='lcr_can1')
    sample_names=df_all_samples_stats[(df_all_samples_stats['exome']==True) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='SNP')]['gt_sample_name'].values
    plt.sca(ax2)
    plt.xticks([r + barWidth for r in range(len(bars1))], sample_names,rotation=90)
    plt.ylabel('SNP F1')
    plt.xlabel('sample name')
    plt.ylim([0.99,1])
    plt.title('Exome')

    ######## Exome Indel ########
    bars1 = df_all_samples_stats[(df_all_samples_stats['exome']==True) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='Indel')]['F1'].values
    # new lcr : WGS,SNP
    bars2 = df_all_samples_stats[(df_all_samples_stats['exome']==True) \
                                 & (df_all_samples_stats['original_lcr']==False) \
                                 & (df_all_samples_stats['type']=='Indel')]['F1'].values

    ax4.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black',  capsize=7, label='v1.4')
    # Create cyan bars
    ax4.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black',  capsize=7, label='lcr_can1')
    sample_names=df_all_samples_stats[(df_all_samples_stats['exome']==True) \
                                 & (df_all_samples_stats['original_lcr']==True) \
                                 & (df_all_samples_stats['type']=='Indel')]['gt_sample_name'].values
    plt.sca(ax4)
    plt.xticks([r + barWidth for r in range(len(bars1))], sample_names,rotation=90)
    plt.ylabel('Indel F1')
    plt.xlabel('sample name')
    plt.ylim([0.92,1])
    plt.title('Exome')


    #fig.legend()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,loc = 'lower center',bbox_to_anchor=(0.5, -0.1))

    fig.suptitle('performace evaluation with new UG-LCR')
    fig.tight_layout()
    plt.show()


def create_candidate(cov, HMER, Tandem, dir_path):
    cmd = f"bedtools multiinter -names COV HMER Tandem \
    -i {cov} \
     {HMER} \
     {Tandem} \
     > {dir_path} / lcr_intersections.bed"
    subprocess.check_call(cmd, shell=True)

    with open(os.path.join(dir_path, "lcr_intersections.bed")) as infile:
        with open(os.path.join(dir_path, "ug_lcr_v2.bed"), 'w') as outfile:
            for line in map(lambda x: x.split(), infile):
                name = line[4].replace(",", "|")
                outfile.write("\t".join((line[0], line[1], line[2], name)))
                outfile.write("\n")
    return os.path.join(dir_path, "ug_lcr_v2.bed")


def extract_bed_for_tag(full_bed_file, tag, GIAB_ref_list,exome):
    """
    generate size summary per lcr tag.
    Parameters
    ----------
    full_bed_file - lcr bed file
    tag - lcr tag (HMER, Tandem, COV)
    GIAB_ref_list - list of GIAB HCR files
    exome - exome bed file

    Returns
    -------
    dataframe with size summary
    """
    hg38_size = 3088269832
    outfile = full_bed_file + '.' + tag + '.bed'
    dirname = os.path.dirname(full_bed_file)

    df = pd.read_csv(exome, header=None, sep='\t')
    exome_size = sum(df[2] - df[1])

    cmd = f"cat {full_bed_file} | grep {tag} > {outfile}"
    subprocess.check_call(cmd, shell=True)

    df = pd.read_csv(outfile, header=None, sep='\t')
    tag_size = sum(df[2] - df[1])

    GIAB_ref_sample = []
    GIAB_intersection = []
    pGIAB_intersection = []

    for GIAB_ref in GIAB_ref_list:
        cmd =f"bedtools intersect -a {outfile} -b {GIAB_ref} | \
            bedtools sort -i - | \
            bedtoolsmerge -i - \
            > {dirname} / tmp.GIAB.bed"
        subprocess.check_call(cmd, shell=True)
        df = pd.read_csv(os.path.join(dirname, 'tmp.GIAB.bed'), header=None, sep='\t')
        GIAB_intersection_size = sum(df[2] - df[1])

        df = pd.read_csv(GIAB_ref, header=None, sep='\t')
        GIAB_size = sum(df[2] - df[1])

        GIAB_intersection_size_p = 100 * (GIAB_intersection_size / GIAB_size)
        GIAB_ref_sample.append(os.path.basename(GIAB_ref).split('_')[0])
        GIAB_intersection.append(GIAB_intersection_size)
        pGIAB_intersection.append(GIAB_intersection_size_p)

    cmd = f"bedtools intersect -a {outfile} -b {exome} | \
        bedtools sort -i - | \
        bedtools merge -i - \
        > {dirname} / tmp.exome.bed"
    subprocess.check_call(cmd, shell=True)
    df = pd.read_csv(os.path.join(dirname, 'tmp.exome.bed'), header=None, sep='\t')
    print(df.shape)
    exome_intersection_size = sum(df[2] - df[1])
    exome_intersection_size_p = 100 * (exome_intersection_size / exome_size)

    hg38_intersection_p = 100 * (tag_size / hg38_size)

    data = {'tag': tag, 'size': tag_size, \
            'exome_intersection': exome_intersection_size, \
            '%exome_intersection': exome_intersection_size_p, \
            '%hg38_intersection': hg38_intersection_p \
            }
    i = 0
    for GIAB_sample in GIAB_ref_sample:
        data[GIAB_sample] = GIAB_intersection[i]
        data['%' + GIAB_sample] = pGIAB_intersection[i]
        i = i + 1

    df_tag = pd.DataFrame(data=data, index=[0])
    return df_tag


def get_stats_for_all_lcr(full_bed_file, GIAB_ref_list,exome, tag='all_lcr'):
    """
    generate size summary for whole lcr candidate.
    Parameters
    ----------
    full_bed_file - lcr bed file
    GIAB_ref_list - list of GIAB HCR files
    exome - exome bed file
    tag - lcr tag (HMER, Tandem, COV)

    Returns
    -------
    dataframe with size summary
    """
    hg38_size = 3088269832
    dirname = os.path.dirname(full_bed_file)

    df = pd.read_csv(exome, header=None, sep='\t')
    exome_size = sum(df[2] - df[1])

    df = pd.read_csv(full_bed_file, header=None, sep='\t')
    tag_size = sum(df[2] - df[1])

    GIAB_ref_sample = []
    GIAB_intersection = []
    pGIAB_intersection = []

    for GIAB_ref in GIAB_ref_list:
        cmd = f"bedtools intersect -a {full_bed_file} -b {GIAB_ref} | \
            bedtools sort -i - | \
            bedtools merge -i - \
            > {dirname} / tmp.GIAB.bed"
        subprocess.check_call(cmd, shell=True)
        df = pd.read_csv(os.path.join(dirname, 'tmp.GIAB.bed'), header=None, sep='\t')
        GIAB_intersection_size = sum(df[2] - df[1])

        df = pd.read_csv(GIAB_ref, header=None, sep='\t')
        GIAB_size = sum(df[2] - df[1])

        GIAB_intersection_size_p = 100 * (GIAB_intersection_size / GIAB_size)
        GIAB_ref_sample.append(os.path.basename(GIAB_ref).split('_')[0])
        GIAB_intersection.append(GIAB_intersection_size)
        pGIAB_intersection.append(GIAB_intersection_size_p)

    plot_GIAB_intersection(GIAB_ref_sample, pGIAB_intersection)

    cmd = f"bedtools intersect -a {full_bed_file} -b {exome} | \
        bedtools sort -i - | \
        bedtools merge -i - \
        > {dirname} / tmp.exome.bed"
    subprocess.check_call(cmd, shell=True)
    df = pd.read_csv(os.path.join(dirname, 'tmp.exome.bed'), header=None, sep='\t')
    exome_intersection_size = sum(df[2] - df[1])
    exome_intersection_size_p = 100 * (exome_intersection_size / exome_size)

    hg38_intersection_p = 100 * (tag_size / hg38_size)

    data = {'tag': tag, 'size': tag_size, \
            'exome_intersection': exome_intersection_size, \
            '%exome_intersection': exome_intersection_size_p, \
            '%hg38_intersection': hg38_intersection_p \
            }
    i = 0
    for GIAB_sample in GIAB_ref_sample:
        data[GIAB_sample] = GIAB_intersection[i]
        data['%' + GIAB_sample] = pGIAB_intersection[i]
        i = i + 1
    df_tag = pd.DataFrame(data=data, index=[0])
    return df_tag


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] + 0.005, round(y[i], 2), ha='center')


def plot_GIAB_intersection(GIAB_ref_sample, pGIAB_intersection):
    """
    plot intersection of lcr with GIAB HCR regions
    Parameters
    ----------
    GIAB_ref_sample - list of GIAB samples
    pGIAB_intersection - corresponding list of %intersection with GIAB HCR sample

    Returns
    -------
    plot of %intersection of lcr with GIAB HCR regions
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 2))

    # width of the bars
    barWidth = 0.3
    bars1 = pGIAB_intersection
    # The x position of bars
    r1 = np.arange(len(bars1))
    # Create blue bars
    ax1.bar(r1, bars1, width=barWidth, color='blue', edgecolor='black', capsize=7, label='v1.4')
    addlabels(r1, bars1)
    plt.sca(ax1)
    plt.xticks([r + barWidth for r in range(len(bars1))], GIAB_ref_sample, rotation=90)
    plt.ylabel('% GIAB HCR intersection ')
    plt.xlabel('sample name')
    plt.ylim([0.7, 1.2])
    # plt.title('WGS')

def create_GIAB_HCR_df():
    """
    download and create GIAB HCR dataframe
    Returns
    -------
    GIAB HCR files location dataframe
    """
    GIAB_HCR_file = ['gs://concordanz/ground-truths-files/HG001_GRCh38_1_22_v4.2.1_benchmark.bed',
                     'gs://concordanz/ground-truths-files/HG002_GRCh38_GIAB_1_22_v4.2.1_benchmark_noinconsistent.bed',
                     'gs://concordanz/ground-truths-files/HG003_GRCh38_GIAB_1_22_v4.2.1_benchmark_noinconsistent.bed',
                     'gs://concordanz/ground-truths-files/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark_noinconsistent.bed',
                     'gs://concordanz/ground-truths-files/HG005_GRCh38_GIAB_1_22_v4.2.1_benchmark.bed',
                     'gs://concordanz/ground-truths-files/HG006_GRCh38_GIAB_1_22_v4.2.1_benchmark.bed',
                     'gs://concordanz/ground-truths-files/HG007_GRCh38_GIAB_1_22_v4.2.1_benchmark.bed'
                     ]

    GIAB_HCR = []
    sample_names = []
    out_folder = '/data/ref_genomes/ground-truths-files/'
    for file in GIAB_HCR_file:
        cmd = f"gsutil cp {file} {out_folder}"
        subprocess.check_call(cmd, shell=True)
        GIAB_HCR.append(pjoin(out_folder, os.path.basename(file)))
        sample_names.append(os.path.basename(file).split('_')[0])
    df_GIAB_HCR = pd.DataFrame({'GIAB_HCR': GIAB_HCR,
                                'sample_names': sample_names})
    return df_GIAB_HCR