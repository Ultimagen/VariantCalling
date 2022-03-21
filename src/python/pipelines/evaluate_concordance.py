#!/env/python
#from simppl.cli import get_parser
import argparse
import sys
from typing import List
import python.modules.pathmagic
from pandas import DataFrame
from python import vcftools
from ugvc.concordance.concordance_utils import calc_accuracy_metrics, calc_recall_precision_curve, read_hdf


def parse_args(argv: List[str]):
    ap = argparse.ArgumentParser(prog="evaluate_concordance.py", description=run.__doc__)
    ap.add_argument("--input_file", help="Name of the input h5 file", type=str, required=True)
    ap.add_argument("--output_prefix", help="Prefix to output files", type=str, required=True)
    ap.add_argument('--dataset_key', help='h5 dataset name, such as chromosome name', default='all')
    ap.add_argument('--ignore_genotype', help='ignore genotype when comparing to ground-truth',
                    action='store_true', default=False)
    ap.add_argument('--ignore_filters', help='comma separated list of filters to ignore', default='HPOL_RUN')
    ap.add_argument('--output_bed', help='output bed files of fp/fn/tp per variant-type', action='store_true',
                    default=False)
    ap.add_argument("--use_for_group_testing",
                    help="Column in the h5 to use for grouping (or generate default groupings)", type=str)

    args = ap.parse_args(argv)
    return args


def run(argv: List[str]):
    """Calculate precision and recall for compared HDF5"""
    args = parse_args(argv)
    ds_key = args.dataset_key
    out_pref = args.output_prefix
    ignore_genotype = args.ignore_genotype
    ignored_filters = args.ignore_filters.split(',')
    output_bed = args.output_bed
    group_testing_column = args.use_for_group_testing

    # comparison dataframes often contain dataframes that we do not want to read
    if ds_key == 'all':
        skip = ['concordance', 'scored_concordance', 'input_args', 'comparison_result']
    else:
        skip = []
    df: DataFrame = read_hdf(args.input_file, key=ds_key, skip_keys=skip)

    # Enable evaluating concordance from vcf without tree-score field
    if all(df['tree_score'].isna()):
        df['tree_score'] = 1

    classify_column = 'classify' if ignore_genotype else 'classify_gt'

    accuracy_df = calc_accuracy_metrics(df, classify_column, ignored_filters, group_testing_column)
    accuracy_df.to_hdf(f'{out_pref}.h5', key="optimal_recall_precision")
    accuracy_df.to_csv(f'{out_pref}.stats.csv', sep=';', index=False)

    recall_precision_curve_df = calc_recall_precision_curve(df, classify_column, ignored_filters)
    recall_precision_curve_df.to_hdf(f'{out_pref}.h5', key="recall_precision_curve")
    if output_bed:
        vcftools.bed_files_output(df, f'{out_pref}.h5', mode='w', create_gt_diff=True)


if __name__ == '__main__':
    run(sys.argv[1:])
