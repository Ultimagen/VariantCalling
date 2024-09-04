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
#    Combine Csv files produced from MethylDackel output files
# ==========================================

# Copyright (c) 2019 Devon Ryan and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==========================================
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="concat_methyldackel_csvs.py",
        description="Concatenate CSV output files of MethylDackel processing into an HDF5 file",
    )
    ap_var.add_argument("--mbias", help="csv summary of MethylDackelMbias", type=str, required=True)
    ap_var.add_argument(
        "--mbias_non_cpg", help="csv summary of MethylDackelMbias in the non-CpG mode", type=str, required=True
    )
    ap_var.add_argument("--merge_context", help="csv summary of MethylDackelMergeContext", type=str, required=True)
    ap_var.add_argument(
        "--merge_context_non_cpg",
        help="csv summary of MethylDackelMergeContext in the non-CpG mode",
        type=str,
        required=True,
    )
    ap_var.add_argument("--per_read", help="csv summary of MethylDackelPerRead", type=str, required=False, default=None)
    ap_var.add_argument("--output", help="Output file basename", type=str, required=True)

    return ap_var.parse_args(argv[1:])


def split_position_hist_desc(df):
    df_per_position = df.loc[df["metric"].str.startswith("PercentMethylationPosition")].copy()
    df_hist = df.loc[df["metric"].str.contains("PercentMethylation_[0-9]+|Coverage_[0-9]+")].copy()
    df_desc = df.loc[~df["metric"].str.contains("_[0-9]+")].copy()
    return df_per_position, df_hist, df_desc


def run(argv: list[str] | None = None):
    "Combine csvs from POST-MethylDackel processing"
    if argv is None:
        argv: list[str] = sys.argv

    args = parse_args(argv)

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running")

    input_dict = {
        "Mbias": args.mbias,
        "MbiasNoCpG": args.mbias_non_cpg,
        "MergeContext": args.merge_context,
        "MergeContextNoCpG": args.merge_context_non_cpg,
        "PerRead": args.per_read,
    }

    h5_output = args.output + ".methyl_seq.applicationQC.h5"
    with pd.HDFStore(h5_output, mode="w") as store:
        for table, input_file in input_dict.items():
            if table == "PerRead" and input_file is None:
                continue
            df = pd.read_csv(input_file)
            df_per_position, df_hist, df_desc = split_position_hist_desc(df)
            tables_to_take = {"per_position": df_per_position, "hist": df_hist, "desc": df_desc}
            for table_ext, df in tables_to_take.items():
                df.set_index(["detail", "metric"], inplace=True)
                df = df.squeeze(axis=1)
                table_name = f"{table}_{table_ext}"
                store.put(table_name, df, format="table", data_columns=True)

        keys_to_convert = pd.Series(
            [
                "Mbias_desc",
                "MbiasNoCpG_desc",
                "MergeContext_desc",
                "MergeContextNoCpG_desc",
                "PerRead_desc",
            ]
        )
        if args.per_read is None:
            keys_to_convert.remove("PerRead_desc")
        store.put("keys_to_convert", pd.Series(keys_to_convert))

    logger.info("Finished")


if __name__ == "__main__":
    run()
