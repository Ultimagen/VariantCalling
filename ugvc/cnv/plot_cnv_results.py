import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import logging
import sys
import seaborn as sns
from ugvc import logger
import warnings
warnings.filterwarnings('ignore')

sns.set_context("talk")
sns.set_style("white")

def smooth_wl (df_reads_count,wl=100000,bin_size=1000) -> pd.DataFrame:
    '''smooth coverage along the genome to a specicic window size (wl) using data with a specific bin size (bin_size)'''
    multiply_factor = wl/bin_size
    df_smooth_wl=pd.DataFrame(columns=['chr','start','end','cov'])
    chr_list=df_reads_count['chr'].unique()
    for chrID in chr_list:
        df_region=df_reads_count[df_reads_count['chr']==chrID]
        start=1
        end=1000*multiply_factor
        while start<df_region['start'].max(): 
            df=df_region[(df_region['start']>=start) & (df_region['end']<=end)]
            cov_value=df.iloc[:,3].median()
            df2 = {'chr': chrID, 'start': start, 'end': end,'cov':cov_value}
            df_smooth_wl = df_smooth_wl.append(df2 ,ignore_index = True )
            start=end+1
            end=end+wl
    return df_smooth_wl

    
def plot_coverage(sample_name,df_chr_graphic,out_directory,df_germline_cov_norm_100K,df_tumor_cov_norm_100K=None) -> str:    
    '''plot coverage along the genome for germline and tumor (if exists) samples'''
    fig = plt.figure(figsize=(20,4))
    marker_size=20

    plt.scatter(df_germline_cov_norm_100K['start_fig'],np.log2(df_germline_cov_norm_100K['cov']),color='blue',alpha=.2,edgecolors='none',s=marker_size)
    if df_tumor_cov_norm_100K is not None:
        plt.scatter(df_tumor_cov_norm_100K['start_fig'],np.log2(df_tumor_cov_norm_100K['cov']),color='orange',alpha=.2,edgecolors='none',s=marker_size)

    previous=0
    xticks=[]
    xticks_labels=[]
    for index, row in df_chr_graphic.iterrows():
        chr = row['chr']
        chr_index = row['start_fig']
        plt.axvline(x=chr_index,color='black',alpha=.5)
        xticks.append(previous+(chr_index-previous)/2)
        xticks_labels.append(chr)
        previous=chr_index
    plt.axvline(x=0,color='black',alpha=.5)

    plt.xticks(xticks, xticks_labels,rotation=60)
    plt.xlabel('location on genome') 
    plt.ylabel('coverage\n(normalized, log scale)') 

    if df_tumor_cov_norm_100K is not None:
        handles = [Rectangle((0,0),0.5,0.5,color=c,ec="k") for c in ['blue','orange']] 
        labels= ['germline','tumor']
    else:
        handles = [Rectangle((0,0),0.5,0.5,color='blue',ec="k")]
        labels= ['germline']
    plt.legend(handles, labels,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([-2,2])
    out_cov_figure = pjoin(out_directory,sample_name+'.CNV.coverage.jpeg')
    plt.savefig(out_cov_figure, dpi=300,bbox_inches="tight")
    return out_cov_figure

def get_x_location_for_fig(df,df_germline_cov_norm_100K) -> pd.DataFrame:
    '''get the x location for the CNV calls in the figure based on the coverage data'''
    start_fig_list = []
    end_fig_list = []
    for index, row in df.iterrows():
        chr=row['chr']
        start=row['start']+1
        end = row['end']+1
       
        start_fig = df_germline_cov_norm_100K[(df_germline_cov_norm_100K['chr']==chr) & (df_germline_cov_norm_100K['start']<=start) & (df_germline_cov_norm_100K['end']>=start)]['start_fig'].values[0]
        end_fig = df_germline_cov_norm_100K[(df_germline_cov_norm_100K['chr']==chr) & (df_germline_cov_norm_100K['start']<=end) & (df_germline_cov_norm_100K['end']>=end)]['start_fig'].values[0]
        start_fig_list.append(start_fig)
        end_fig_list.append(end_fig)
    df['start_fig']=start_fig_list
    df['end_fig']=end_fig_list
    return df

def plot_amp_del_cnv_calls(df_chr_graphic,out_directory,sample_name,df_DUP=None,df_DEL=None,df_gt_DUP=None,df_gt_DEL=None) -> str:
    '''plot CNV calls showing duplication and deletions along the genome with comparison to ground truth.'''
    fig = plt.figure(figsize=(20,2))
    marker_size=20

    gt = False
    if df_gt_DEL is not None:
        plt.plot((df_gt_DEL['start_fig'], df_gt_DEL['end_fig']), (0, 0), 'red')
        gt=True
    if df_gt_DUP is not None:
        plt.plot((df_gt_DUP['start_fig'], df_gt_DUP['end_fig']), (0, 0), 'green')
        gt=True
    germline = False
    if df_DEL is not None:
        plt.plot((df_DEL['start_fig'], df_DEL['end_fig']), (0.5, 0.5), 'red')
        germline = True
    if df_DUP is not None:
        plt.plot((df_DUP['start_fig'], df_DUP['end_fig']), (0.5, 0.5), 'green')
        germline = True

    previous=0
    xticks=[]
    xticks_labels=[]
    for index, row in df_chr_graphic.iterrows():
        chr = row['chr']
        chr_index = row['start_fig']
        plt.axvline(x=chr_index,color='black',alpha=.5)
        xticks.append(previous+(chr_index-previous)/2)
        xticks_labels.append(chr)
        previous=chr_index
    plt.axvline(x=0,color='black',alpha=.5)

    plt.xticks(xticks, xticks_labels,rotation=60)
  
    if gt and germline:
        plt.yticks([0,0.5],['Ground Truth','UG calls'])
    elif gt and not germline:
        plt.yticks([0],['Ground Truth'])
    elif germline and not gt:
        plt.yticks([0.5],['UG calls'])
    
    plt.xlabel('location on genome') 
    handles = [Rectangle((0,0),0.5,0.5,color=c,ec="k") for c in ['green','red']] 
    labels= ['Duplication','Deletion']
    plt.legend(handles, labels,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([-0.5,1])

    out_calls_figure = pjoin(out_directory,sample_name+'.dup_del.calls.jpeg')
    plt.savefig(out_calls_figure, dpi=300,bbox_inches="tight")
    return out_calls_figure

def plot_cnv_calls(sample_name,out_directory,df_chr_graphic,df_germline_cov_norm_100K,df_DUP=None,df_DEL=None) -> str:
    ''' plot the copy number along the genome'''
    if df_DUP is None and df_DEL is None:
        return None
    else:
        df_calls = pd.concat([df_DUP,df_DEL])
        get_x_location_for_fig(df_calls,df_germline_cov_norm_100K)
        df_calls['width']=df_calls['end']-df_calls['start']
        df_calls['width_fig']=df_calls['end_fig']-df_calls['start_fig']

        fig = plt.figure(figsize=(20,2))
        marker_size=20

        plt.plot((df_calls['start_fig'],df_calls['end_fig']),(df_calls['copy-number'],df_calls['copy-number']),'black')

        previous=0
        xticks=[]
        xticks_labels=[]
        for index, row in df_chr_graphic.iterrows():
            chr = row['chr']
            chr_index = row['start_fig']
            plt.axvline(x=chr_index,color='black',alpha=.5)
            xticks.append(previous+(chr_index-previous)/2)
            xticks_labels.append(chr)
            previous=chr_index
        plt.axvline(x=0,color='black',alpha=.5)

        plt.axhline(y=3,color='grey',alpha=.5)

        plt.xticks(xticks, xticks_labels,rotation=60)
        plt.yticks([0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8])
        plt.xlabel('location on genome') 
        plt.ylabel('copy-number') 

        handles = [Rectangle((0,0),0.5,0.5,color=c,ec="k") for c in ['black']] 
        labels= ['copy-number']
        plt.legend(handles, labels,loc='center left', bbox_to_anchor=(1, 0.5))

        #plt.ylim([0,8])
        out_calls_figure = pjoin(out_directory,sample_name+'.CNV.calls.jpeg')
        plt.savefig(out_calls_figure, dpi=300,bbox_inches="tight")
        return out_calls_figure

def run(argv):
    """
    Runs the plot_cnv_results.py script to generate CNV calling results description in figures.
    input arguments:
    --germline_coverage: input bed file holding the germline coverage along the genome.
    --tumor_coverage: input bed file holding the tumor coverage along the genome.
    --duplication_cnv_calls: input tsv file holding DUP CNV calls in the following format: <chr><start><end><copy-number>
    --deletion_cnv_calls: input tsv file holding DEL CNV calls in the following format: <chr><start><end><copy-number>
    --gt_duplication_cnv_calls: input tsv file holding ground truth DUP CNV calls in the following format: <chr><start><end><copy-number>
    --gt_deletion_cnv_calls: input tsv file holding ground truth DEL CNV calls in the following format: <chr><start><end><copy-number>
    --out_directory: output directory
    --sample_name: sample name    
    output files:
    coverage plot: <sample_name>.CNV.coverage.jpeg 
        shows normalized (log scale) coverage along the genome for the germline (and tumor) samples. 
    duplications and deletions figure : <sample_name>.dup_del.calls.jpeg
        CNV calls plot showing duplication and deletions along the genome with comparison to ground truth.
    copy-number figure: <sample_name>.CNV.calls.jpeg
        shows the copy number along the genome.
    """
    parser = argparse.ArgumentParser(
        prog="plot_cnv_results.py", description="plot CNV calling results"
    )
    
    parser.add_argument("--germline_coverage", help="input bed file holding the germline coverage along the genome", required=True, type=str)
    parser.add_argument("--tumor_coverage", help="input bed file holding the tumor coverage along the genome", required=False, type=str)
    parser.add_argument("--duplication_cnv_calls", help="input tsv file holding DUP CNV calls in the following format: <chr><start><end><copy-number>", required=False, type=str)
    parser.add_argument("--deletion_cnv_calls", help="input tsv file holding DEL CNV calls in the following format: <chr><start><end><copy-number>", required=False, type=str)
    parser.add_argument("--gt_duplication_cnv_calls", help="input tsv file holding ground truth DUP CNV calls in the following format: <chr><start><end><copy-number>", required=False, type=str)
    parser.add_argument("--gt_deletion_cnv_calls", help="input tsv file holding ground truth DEL CNV calls in the following format: <chr><start><end><copy-number>", required=False, type=str)
    parser.add_argument("--out_directory", help="output directory", required=True, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity",help="Verbosity: ERROR, WARNING, INFO, DEBUG",required=False,default="INFO",)

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    is_somatic = False
    if args.tumor_coverage:
        is_somatic = True
        logger.info("somatic CNV results will be plotted")
    else:    
        logger.info("germline CNV results will be plotted")

    #make output directory 
    if os.path.isdir(args.out_directory):
        logger.info("out directory exists, will output files into this directory: %s", args.out_directory)
    else:
        os.makedirs(args.out_directory)
        logger.info("creating out directory : %s", args.out_directory)
    
    #########################
    ##### plot coverage #####
    #########################
        
    #load & normalize coverage
    df_germline_cov = pd.read_csv(args.germline_coverage,header=None,sep='\t')
    df_germline_cov.columns=['chr','start','end','cov']
    df_germline_cov['norm_cov'] = df_germline_cov['cov']/df_germline_cov['cov'].median()
    df_germline_cov['log_norm_cov'] = np.log2(df_germline_cov['norm_cov'])
    df_germline_cov_norm_100K=smooth_wl(df_germline_cov[['chr','start','end','norm_cov']],100000)
    df_germline_cov_norm_100K['start_fig']=list(range(len(df_germline_cov_norm_100K)))
    
    df_chr_graphic = pd.DataFrame({'chr':df_germline_cov_norm_100K.groupby(['chr'])['start_fig'].max().index,
                              'start_fig':df_germline_cov_norm_100K.groupby(['chr'])['start_fig'].max().values})
    df_chr_graphic['chr_num']=df_chr_graphic['chr'].str.replace('chr', '', regex=True)
    df_chr_graphic['chr_num'] = df_chr_graphic['chr_num'].str.replace('X', '23', regex=True)
    df_chr_graphic['chr_num'] = df_chr_graphic['chr_num'].astype(int)
    df_chr_graphic = df_chr_graphic.sort_values(by=['chr_num'])

    if is_somatic:
        df_tumor_cov = pd.read_csv(args.tumor_coverage,header=None,sep='\t')
        df_tumor_cov.columns=['chr','start','end','cov']
        df_tumor_cov['norm_cov'] = df_tumor_cov['cov']/df_tumor_cov['cov'].median()
        df_tumor_cov['log_norm_cov'] = np.log2(df_tumor_cov['norm_cov'])
        df_tumor_cov_norm_100K=smooth_wl(df_tumor_cov[['chr','start','end','norm_cov']],100000)
        df_tumor_cov_norm_100K['start_fig']=list(range(len(df_tumor_cov_norm_100K)))
        out_cov_figure = plot_coverage(args.sample_name,df_chr_graphic,args.out_directory,df_germline_cov_norm_100K,df_tumor_cov_norm_100K)
    else:
        out_cov_figure = plot_coverage(args.sample_name,df_chr_graphic,args.out_directory,df_germline_cov_norm_100K)
    
    ##########################
    ##### plot CNV calls #####
    ##########################
    #load UG calls
    if args.duplication_cnv_calls:
        df_DUP = pd.read_csv(args.duplication_cnv_calls,sep='\t',header=None)
        df_DUP.columns=['chr','start','end','copy-number']
        df_DUP = get_x_location_for_fig(df_DUP,df_germline_cov_norm_100K)
    else:
        df_DUP = None
    if args.deletion_cnv_calls:
        df_DEL = pd.read_csv(args.deletion_cnv_calls,sep='\t',header=None)
        df_DEL.columns=['chr','start','end','copy-number']
        df_DEL = get_x_location_for_fig(df_DEL,df_germline_cov_norm_100K)
    else:
        df_DEL = None
    if args.gt_duplication_cnv_calls:
        df_gt_DUP = pd.read_csv(args.gt_duplication_cnv_calls,sep='\t',header=None)
        df_gt_DUP.columns=['chr','start','end','copy-number']
        df_gt_DUP = get_x_location_for_fig(df_gt_DUP,df_germline_cov_norm_100K)
    else:
        df_gt_DUP = None
    if args.gt_deletion_cnv_calls:
        df_gt_DEL = pd.read_csv(args.gt_deletion_cnv_calls,sep='\t',header=None)
        df_gt_DEL.columns=['chr','start','end','copy-number']
        df_gt_DEL = get_x_location_for_fig(df_gt_DEL,df_germline_cov_norm_100K)
    else:
        df_gt_DEL = None
    out_dup_del_calls_figure = plot_amp_del_cnv_calls(df_chr_graphic,args.out_directory,args.sample_name,df_DUP,df_DEL,df_gt_DUP,df_gt_DEL)
    out_cnv_calls_figure = plot_cnv_calls(args.sample_name,args.out_directory,df_chr_graphic,df_germline_cov_norm_100K,df_DUP,df_DEL)

    logger.info("output files:")
    logger.info(out_cov_figure)
    logger.info(out_dup_del_calls_figure)
    logger.info(out_cnv_calls_figure)
    


if __name__ == "__main__":
    run(sys.argv)