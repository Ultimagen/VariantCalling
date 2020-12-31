import sys
import glob
import numpy as np
import pandas as pd
import h5py as h5
from os.path import join as pjoin
import os
import json

home = os.environ['HOME']
sys.path.append(pjoin(home, 'proj/BioinfoResearch/VariantCalling/src/'))
sys.path.append(pjoin(home, 'proj/Base-calling/ContextResearch/'))
from python.pipelines import vc_pipeline_utils
workdir = pjoin(home, "proj/BioinfoResearch/work/")
import argparse
ap = argparse.ArgumentParser(
    prog="collect_existing_picard_metrics.py", description="Collect picard metrics in h5 file")

ap.add_argument('--metric_files', nargs='+',help="comma seperated list of picard metric files")
ap.add_argument("--coverage_h5", help='Coverage h5 File',
                required=True, type=str)
ap.add_argument("--output_h5", help='Aggregated Metrics h5 file',
                required=True, type=str)

args = ap.parse_args()

for metric_file in args.metric_files: #glob.glob(pjoin(workdir,"*metrics")):
    if os.path.getsize(metric_file) > 0:
         metric_class,stats,histogram = vc_pipeline_utils.parse_cvg_metrics(metric_file)
         metric_class = metric_class[metric_class.find("$")+1:]
         stats.to_hdf(args.output_h5,key=metric_class, mode="a")
         if histogram is not None:
             histogram.to_hdf(args.output_h5, key= "histogram_" + metric_class, mode="a")

cvg_h5_stats = pd.read_hdf(args.coverage_h5, key="stats")
cvg_h5_histogram = pd.read_hdf(args.coverage_h5, key="histogram")
cvg_df=pd.DataFrame(cvg_h5_stats.loc[("5th percentile","10th percentile","median coverage","median coverage (normalized to median genome coverage)","% bases with coverage >= 10x","% bases with coverage >= 20x"),:])
cvg_df=cvg_df.rename(index={"median coverage (normalized to median genome coverage)":"median_coverage_normalized","median coverage": "median_coverage","5th percentile":"5th_percentile","10th percentile": "10th_percentile","% bases with coverage >= 20x":"percent_bases_above_20x","% bases with coverage >= 10x":"percent_bases_above_10x"})
cvg_df_unstacked=pd.DataFrame(cvg_df.unstack(level=0)).T
cvg_df_unstacked.to_hdf(args.output_h5,key="stats_coverage", mode="a")
cvg_h5_histogram.to_hdf(args.output_h5,key="histogram_coverage", mode="a")
