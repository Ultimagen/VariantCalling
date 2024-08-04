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
#    Convert BICSEQ2 default output to bed file with Copy Number annotations.
# CHANGELOG in reverse chronological order

import argparse
import logging
import os
import sys
import pandas as pd
import warnings

from ugvc import logger

warnings.filterwarnings("ignore")

def check_path(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        logger.info("creating out directory : %s", path)

def run(argv):
    """
    Conversion of bicseq2 output text file to BED format :
    """
    parser = argparse.ArgumentParser(
        prog="bicseq2_post_processing.py", description="Conversion of bicseq2 output text file to BED format."
    )
    parser.add_argument("--input_bicseq2_txt_file", help="input bicseq2 default output file", required=True, type=str)
    parser.add_argument(
        "--ratio_DUP_cutoff",
        help="log2.copyRatio cutoff to annotate as DUPLICATION",
        required=False,
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--ratio_DEL_cutoff",
        help="log2.copyRatio cutoff to annotate as DELETION",
        required=False,
        type=float,
        default=-0.25,
    )
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    prefix = ""
    if args.out_directory:
        prefix = args.out_directory
        prefix = prefix.rstrip("/") + "/"

    df_bicseq_results = pd.read_csv(args.input_bicseq2_txt_file, sep='\t')

    del_cutoff = args.ratio_DEL_cutoff
    dup_cutoff = args.ratio_DUP_cutoff

    df_bicseq_results['CNV'] = 'NEU'
    df_bicseq_results.loc[
        (df_bicseq_results['log2.copyRatio'] < del_cutoff), 'CNV'] = 'DEL'
    df_bicseq_results.loc[
        (df_bicseq_results['log2.copyRatio'] > dup_cutoff), 'CNV'] = 'DUP'

    pre, ext = os.path.splitext(os.path.basename(args.input_bicseq2_txt_file))
    out_file = prefix + pre+'.bed'

    df_bicseq_results[['chrom', 'start', 'end', 'CNV']].to_csv(out_file, sep='\t',header=None, index=False)

    logger.info("output file:")
    logger.info(out_file)

if __name__ == "__main__":
    run(sys.argv)
