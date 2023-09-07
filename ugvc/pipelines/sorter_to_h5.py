from __future__ import annotations

import argparse

from ugvc.utils.sorter_to_h5 import sorter_to_h5


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert sort output (csv, json file) into an aggregated h5 file\n", description=run.__doc__
    )
    parser.add_argument(
        "-c",
        "--input_csv_file",
        type=str,
        required=True,
        help="Path to the input csv file",
    )
    parser.add_argument(
        "-j",
        "--input_json_file",
        type=str,
        required=True,
        help="Path to the input json file",
    )
    parser.add_argument(
        "-m",
        "--metric_mapping_file",
        type=str,
        required=True,
        help="Path to the metric mapping file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Aggregate sorter metrics into an h5 file of the format: aggregated_metrics.h5"""
    args_in = __parse_args(argv)
    sorter_to_h5(
        input_csv_file=args_in.input_csv_file,
        input_json_file=args_in.input_json_file,
        metric_mapping_file=args_in.metric_mapping_file,
        output_dir=args_in.output_dir,
    )
