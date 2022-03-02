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

def preprocess_columns(dataframe):
    """Handle multiIndex/ hierarchical .h5 - concatenate the columns for using it as single string in JSON."""
    if hasattr(dataframe, 'columns'):
        MULTI_INDEX_SEPARATOR = "___"
        if isinstance(dataframe.columns, pd.core.indexes.multi.MultiIndex):
            dataframe.columns = [MULTI_INDEX_SEPARATOR.join(col).rstrip(
                MULTI_INDEX_SEPARATOR) for col in dataframe.columns.values]

def add_h5_to_hdf(input_h5_name, output_h5_name, output_report_key_prefix):
    with pd.HDFStore(input_h5_name, 'r') as hdf:
        hdf_keys = hdf.keys()
        for report_key in hdf_keys:
            report_h5_pd = pd.read_hdf(
                input_h5_name, key=report_key)
            report_h5_pd_df = pd.DataFrame(report_h5_pd)
            report_h5_unstacked = pd.DataFrame(report_h5_pd_df.unstack(level=list(range(report_h5_pd_df.index.nlevels)))).T
            report_h5_unstacked.to_hdf(
                output_h5_name, key=output_report_key_prefix + report_key, mode="a")

args = ap.parse_args()
if args.metric_files is not None:
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
    add_h5_to_hdf(args.short_report_h5, args.output_h5, "short_report_")

if args.extended_report_h5 is not None:
    add_h5_to_hdf(args.extended_report_h5, args.output_h5, "extended_report_")

if args.contamination_stdout is not None:
    contamination_df = pd.DataFrame(
        pd.Series(data=[float(args.contamination_stdout)], index=["contamination"])).T
    contamination_df.to_hdf(args.output_h5, key="contamination", mode="a")

if args.no_gt_report_h5 is not None:
    add_h5_to_hdf(args.no_gt_report_h5, args.output_h5, "no_gt_report_")
