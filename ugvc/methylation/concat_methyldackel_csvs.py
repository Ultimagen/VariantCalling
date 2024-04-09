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
        description="Concatenate CSV output files of MethylDackel processing.",
    )
    ap_var.add_argument(
        "--mbias", help="csv summary of MethylDackelMbias", type=str, required=True
    )
    ap_var.add_argument(
        "--mbias_non_cpg", help="csv summary of MethylDackelMbias in the non-CpG mode", type=str, required=True
    )
    ap_var.add_argument(
        "--merge_context", help="csv summary of MethylDackelMergeContext", type=str, required=True
    )
    ap_var.add_argument(
        "--merge_context_non_cpg",
        help="csv summary of MethylDackelMergeContext in the non-CpG mode",
        type=str,
        required=True,
    )
    ap_var.add_argument(
        "--per_read", help="csv summary of MethylDackelPerRead", type=str, required=True
    )
    ap_var.add_argument(
        "--output", help="Output file basename", type=str, required=True
    )

    return ap_var.parse_args(argv[1:])


def run(argv: list[str]):
    "Combine csvs from POST-MethylDackel processing"
    args = parse_args(argv)
    # print(f"Processing file {args}")

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running")

    try:
        # check if input files exist

        df_mbias = pd.read_csv(args.mbias)
        df_mbias["table"] = "Mbias"
        df_mbias_non_cpg = pd.read_csv(args.mbias_non_cpg)
        df_mbias_non_cpg["table"] = "MbiasNoCpG"
        df_merge_context = pd.read_csv(args.merge_context)
        df_merge_context["table"] = "MergeContext"
        df_merge_context_non_cpg = pd.read_csv(args.merge_context_non_cpg)
        df_merge_context_non_cpg["table"] = "MergeContextNoCpG"
        df_per_read = pd.read_csv(args.per_read)
        df_per_read["table"] = "PerRead"

        df_csv_output = pd.concat([df_mbias, df_mbias_non_cpg, df_merge_context, df_merge_context_non_cpg, df_per_read])
        # parse to create more readable columns
        temp = df_csv_output["metric"].str.split("_", n=1, expand=True)
        temp.columns = ["measure", "bin"]
        df_csv_output = pd.concat([df_csv_output, temp], axis=1)

        df_csv_output["measure"] = df_csv_output["measure"].str.replace(
            r"PercentMethylation$", "Percent Methylation", regex=True
        )
        df_csv_output["measure"] = df_csv_output["measure"].str.replace(
            r"MethylationPosition", " Methylation Position", regex=True
        )
        df_csv_output["measure"] = df_csv_output["measure"].str.replace(r"TotalCpGs", "Total CpGs", regex=True)

        # print out combined CSVs into one CSV file
        # ==========================================================================================
        out_file_name = args.output + ".csv"
        df_csv_output.to_csv(out_file_name, index=False, na_rep="NULL", header=True, encoding="utf-8")

        # ==============
        # Fin
        # ==============

    except Exception as err:
        exc_info = sys.exc_info()
        logger.exception(*exc_info)
        logger.error("Combining CSV files run: failed")
        raise err


if __name__ == "__main__":
    run(sys.argv[1:])
