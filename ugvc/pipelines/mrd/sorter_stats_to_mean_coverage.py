from __future__ import annotations

import argparse

import numpy as np

from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="annotate featuremap", description=run.__doc__)
    parser.add_argument(
        "-i",
        "--sorter-stats-json",
        type=str,
        required=True,
        help="input sorter stats JSON file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Path of output file",
    )
    return parser.parse_args(argv[1:])


def sorter_stats_to_mean_coverage(sorter_stats_json: str, output_file: str):
    """
    Read mean coverage metric from sorter JSON file and write the up-rounded result to a file.

    Parameters
    ----------
    sorter_stats_json : str
        Path to Sorter statistics JSON file.
    output_file : str
        Path to output file.

    """
    mean_cvg = np.ceil(read_effective_coverage_from_sorter_json(sorter_stats_json)[0])
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{mean_cvg:.0f} ")


def run(argv: list[str]):
    """Read effective coverage metrics from sorter JSON file and write the up-rounded result to a file."""
    args = parse_args(argv)

    sorter_stats_to_mean_coverage(
        sorter_stats_json=args.sorter_stats_json,
        output_file=args.output_file,
    )
