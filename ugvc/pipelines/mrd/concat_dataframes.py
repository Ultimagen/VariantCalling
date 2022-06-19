#!/env/python

# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Concatenate dataframes
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

import pandas as pd
from joblib import Parallel, delayed


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="concat_dataframes", description=run.__doc__)
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="input featuremap files",
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

    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Concat featuremap pandas dataframe created on different intevals"""
    args = __parse_args(argv)
    concat_dataframes(
        dataframes=args.input,
        outfile=args.output,
        n_jobs=args.jobs,
    )
    print("DONE")


def concat_dataframes(dataframes: list, outfile: str, n_jobs: int = 1):
    df = pd.concat(Parallel(n_jobs=n_jobs)(delayed(pd.read_parquet)(f) for f in dataframes))
    df = df.sort_index()
    df.to_parquet(outfile)
    return df
