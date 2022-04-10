#!/env/python
import argparse
from typing import List

import pandas as pd
from joblib import Parallel, delayed


def __parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="concat_dataframes", description=run.__doc__)
    parser.add_argument(
        "input", nargs="+", type=str, help="input featuremap files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="""Path to which output dataframe will be written, if None a file with the same name as input and
        ".parquet" extension will be created""",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel (default 1, -1 for max)",
    )

    return parser.parse_args(argv)


def run(argv: List[str]):
    """Concat featuremap pandas dataframe created on different intevals"""
    args = __parse_args(argv)
    concat_dataframes(
        dataframes=args.input, outfile=args.output, n_jobs=args.jobs,
    )
    print("DONE")


def concat_dataframes(dataframes: list, outfile: str, n_jobs: int = 1):
    df = pd.concat(
        Parallel(n_jobs=n_jobs)(delayed(pd.read_parquet)(f) for f in dataframes)
    )
    df = df.sort_index()
    df.to_parquet(outfile)
    return df
