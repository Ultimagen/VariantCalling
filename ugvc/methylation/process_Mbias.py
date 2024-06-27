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

from ugvc.methylation.methyldackel_utils import calc_distrib_per_strand, get_dict_from_dataframe


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(
        prog="process_Mbias.py",
        description="Process MethylDackel information along"
        "reads to calculate methylation on strands: OT, OB and CTOB, CTOB if prepared ",
    )
    ap_var.add_argument("--input", help="MethylDackel Mbias input file", type=str, required=True)
    ap_var.add_argument("--output", help="Output file basename", type=str, required=True)
    ap_var.add_argument("--noCpG", help="use this flag to tag Mbias call on no-CpG cytosines", type=bool, required=False)

    return ap_var.parse_args(argv[1:])


def run(argv: list[str] | None = None):
    """POST-MethylDackel Mbias processing"""
    if argv is None:
        argv: list[str] = sys.argv

    args = parse_args(argv)
    # print(f"Processing file {args}")

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running..........")

    try:
        # import Mbais file
        in_file_name = args.input
        df_mbias_input = pd.read_csv(in_file_name, sep="\t")
        df_mbias_input["PercentMethylation"] = df_mbias_input["nMethylated"] / (
            df_mbias_input["nMethylated"] + df_mbias_input["nUnmethylated"]
        )
        cols = ["Read", "nMethylated", "nUnmethylated"]
        df_mbias_input.drop(columns=cols, inplace=True)

        # create dataframe from input files
        # ===================================================================
        df_csv_output = calc_distrib_per_strand(df_mbias_input)

        # print to CSV file
        # ==========================================================================================
        out_file_name = args.output + ".csv"
        df_csv_output.to_csv(out_file_name, index=False, na_rep="NULL", header=True, encoding="utf-8")

        # create dictionary for writing entire data to json file, use above function
        # ==========================================================================================
        dict_json_output = {}
        for strand in df_csv_output["detail"].unique():
            temp_dict = get_dict_from_dataframe(df_csv_output, strand)
            dict_json_output.update(temp_dict)

        # print to JSON file
        # ==========================================================================================
        metric_title = "Mbias"
        if args.noCpG:
            metric_title = "MbiasNoCpG"

        out_json = {"metrics": {}}
        out_json["metrics"] = {metric_title : dict_json_output}
        out_file_name = args.output + ".json"
        with open(out_file_name, "w", encoding="utf-8") as file_handler:
            json.dump(out_json, file_handler, indent=2, default=str)

        # ===================================================================
        # Fin
        # ===================================================================

    except Exception as err:
        exc_info = sys.exc_info()
        logger.exception(*exc_info)
        logger.error("Processing MethylDackel Mbias run: failed")
        raise err


if __name__ == "__main__":
    run()
