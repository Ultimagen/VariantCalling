import sys
import glob
import numpy as np
import pandas as pd
import h5py as h5
from os.path import join as pjoin
import os
import json
import pathmagic
from python.pipelines import vc_pipeline_utils
import argparse

ap = argparse.ArgumentParser(
    prog="collect_existing_picard_metrics.py", description="Collect picard metrics in h5 file")
ap.add_argument('--metric_files', nargs='+', help="comma seperated list of picard metric files",
                required=False)
ap.add_argument("--coverage_h5", help='Coverage h5 File',
                required=False, type=str)
ap.add_argument("--short_report_h5", help='Short report h5 file',
                required=False, type=str)
ap.add_argument("--extended_report_h5", help='Extended report h5 file',
                required=False, type=str)
ap.add_argument("--no_gt_report_h5", help='No ground truth report h5 file',
                required=False, type=str)
ap.add_argument("--output_h5", help='Aggregated Metrics h5 file',
                required=False, type=str)
ap.add_argument("--contamination_stdout", help='Rate of Contamination',
                required=False, type=str)


args = ap.parse_args()

for metric_file in args.metric_files:
    if os.path.getsize(metric_file) > 0:
        metric_class, stats, histogram = vc_pipeline_utils.parse_cvg_metrics(
            metric_file)
        metric_class = metric_class[metric_class.find("$")+1:]
        stats.to_hdf(args.output_h5, key=metric_class, mode="a")
        if histogram is not None:
            histogram.to_hdf(
                args.output_h5, key="histogram_" + metric_class, mode="a")

if args.coverage_h5 is not None:
    cvg_h5_histogram = pd.read_hdf(args.coverage_h5, key="histogram")
    cvg_df = pd.read_hdf(args.coverage_h5, key="stats")
    cvg_df_unstacked = pd.DataFrame(cvg_df.unstack(level=0)).T
    cvg_df_unstacked.to_hdf(args.output_h5,key="stats_coverage", mode="a")
    cvg_h5_histogram.to_hdf(args.output_h5,key="histogram_coverage", mode="a")

if args.short_report_h5 is not None:
    with pd.HDFStore(args.short_report_h5, 'r') as hdf:
        hdf_keys = hdf.keys()
        for report_key in hdf_keys:
            short_report_h5_pd = pd.read_hdf(
                args.short_report_h5, key=report_key)
            short_report_h5_pd_df = pd.DataFrame(short_report_h5_pd)
            short_report_h5_unstacked = pd.DataFrame(
                short_report_h5_pd_df.unstack(level=0)).T
            short_report_h5_unstacked.to_hdf(
                args.output_h5, key="short_report_" + report_key, mode="a")
# if args.extended_report_h5 is not None:
#     with pd.HDFStore(args.extended_report_h5,'r') as hdf:
#         hdf_keys = hdf.keys()
#         for report_key in hdf_keys:
#             extended_report_h5_pd = pd.read_hdf(args.extended_report_h5, key= report_key)
#             extended_report_h5_pd_df = pd.DataFrame(extended_report_h5_pd.unstack(level=list(range(extended_report_h5_pd.index.nlevels))))
#             extended_report_h5_pd_df.index = extended_report_h5_pd_df.index.to_flat_index()
#             extended_report_h5_pd_df = extended_report_h5_pd_df.T
#             extended_report_h5_pd_df.to_hdf(args.output_h5, key="extended_report_" + report_key, mode="a")
if args.contamination_stdout is not None:
    contamination_df = pd.DataFrame(
        pd.Series(data=[float(args.contamination_stdout)], index=["contamination"])).T
    contamination_df.to_hdf(args.output_h5, key="contamination", mode="a")

if args.no_gt_report_h5 is not None:
    with pd.HDFStore(args.no_gt_report_h5, 'r') as hdf:
        hdf_keys = hdf.keys()
        for report_key in hdf_keys:
            no_gt_report_h5_pd = pd.read_hdf(
                args.no_gt_report_h5, key=report_key)
            no_gt_report_h5_pd_df = pd.DataFrame(no_gt_report_h5_pd)
            no_gt_report_h5_unstacked = pd.DataFrame(
                no_gt_report_h5_pd_df.unstack(level=0)).T
            no_gt_report_h5_unstacked.to_hdf(
                args.output_h5, key="no_gt_report_" + report_key, mode="a")