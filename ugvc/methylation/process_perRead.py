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
#    Process MethylDackel output file
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
import json
import logging
import sys

import pandas as pd

from ugvc.methylation.methyldackel_utils import calc_percent_methylation, calc_TotalCpGs, get_dict_from_dataframe


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="process_perRead.py",
        description="Process MethylDackel perRead "
        " Calculate percent methylation and coverage pre reads."
        " Create CSV and JSON files as output.",
    )
    ap_var.add_argument("--input", help="MethylDackel perRead file", type=str, required=True)
    ap_var.add_argument("--output", help="Output file basename", type=str, required=True)

    return ap_var.parse_args(argv[1:])


def run(argv: list[str] | None = None):
    "POST-MethylDackel perRead processing"
    if argv is None:
        argv: list[str] = sys.argv

    args = parse_args(argv)

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running")

    try:

        # import file
        in_file_name = args.input
        col_names = ["read_name", "chr", "start", "PercentMethylation", "TotalCpGs"]
        df_perRead = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)
        df_perRead.drop(columns="read_name", inplace=True)
        df_perRead.dropna(inplace=True)

        # get dataframe of % methylation, no. of CpGs per reads for each genome in MethylDackel output file
        # ==========================================================================================
        df_pcnt_meth = calc_percent_methylation("PercentMethylation", df_perRead, True)
        df_total_cpgs = calc_TotalCpGs("TotalCpGs", df_perRead)
        df_csv_output = pd.concat([df_pcnt_meth, df_total_cpgs], axis=0, ignore_index=True)

        # print to CSV file
        # ==========================================================================================
        out_file_name = args.output + ".csv"
        df_csv_output.to_csv(out_file_name, index=False, na_rep="NULL", header=True, encoding="utf-8")

        # create dictionary for writing entire data to json file, use above function
        # ===================================================================
        dict_json_output = {}
        for detail in df_csv_output["detail"].unique():
            temp_dict = get_dict_from_dataframe(df_csv_output, detail)
            dict_json_output.update(temp_dict)

        # print to JSON file
        # ==========================================================================================
        out_json = {"metrics": {}}
        out_json["metrics"] = {"PerRead": dict_json_output}

        out_file_name = args.output + ".json"
        with open(out_file_name, "w", encoding="utf-8") as file_handler:
            json.dump(out_json, file_handler, indent=2, default=str)

        # ===================================================================
        # Fin
        # ===================================================================

    except Exception as err:
        exc_info = sys.exc_info()
        logger.exception(*exc_info)
        logger.error("Processing MethylDackel perRead run: failed")
        raise err


if __name__ == "__main__":
    run()
