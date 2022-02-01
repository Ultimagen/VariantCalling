#!/env/python
import argparse

from pandas import DataFrame

from python import vcftools
from ugvc.concordance.concordance_utils import calc_accuracy_metrics, calc_recall_precision_curve, read_hdf


def parse_args():
    ap = argparse.ArgumentParser(prog="evaluate_concordance.py",
                                 description="Calculate precision and recall for compared HDF5 ")
    ap.add_argument("--input_file", help="Name of the input h5 file", type=str, required=True)
    ap.add_argument("--output_prefix", help="Prefix to output files", type=str, required=True)
    ap.add_argument('--dataset_key', help='h5 dataset name, such as chromosome name', default='all')
    ap.add_argument('--ignore_genotype', help='ignore genotype when comparing to ground-truth',
                    action='store_true', default=False)

    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    ds_key = args.dataset_key
    out_pref = args.output_prefix
    ignore_genotype = args.ignore_genotype
    df: DataFrame = read_hdf(args.input_file, key=ds_key)

    classify_column = 'classify' if ignore_genotype else 'classify_gt'

    accuracy_df = calc_accuracy_metrics(df, classify_column)
    accuracy_df.to_hdf(f'{out_pref}.h5', key="optimal_recall_precision")
    accuracy_df.to_csv(f'{out_pref}.stats.tsv', sep='\t', index=False)

    recall_precision_curve_df = calc_recall_precision_curve(df, classify_column)
    recall_precision_curve_df.to_hdf(f'{out_pref}.h5', key="recall_precision_curve")
    vcftools.bed_files_output(df, out_pref, mode='w', create_gt_diff=True)


if __name__ == '__main__':
    main()
