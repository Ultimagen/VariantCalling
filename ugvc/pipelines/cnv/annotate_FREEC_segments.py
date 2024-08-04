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
#    Annotate segments outputted from controlFREEC as gain/loss/neutral.
# CHANGELOG in reverse chronological order

import argparse
import logging
import sys
import os
import pandas as pd
from ugvc import logger
    
def run(argv):
    """
    Given a segments file outputed from controlFREEC, this script will annotate each segment as gain/loss/neutral. 
    The annotation is based on the median fold-change of the segment.
    The script requires the following inputs:
    1. segments file  2. gain cutoff  3. loss cutoff
    output:
    - annotated segments file
    - CNVs bed file
    """
    parser = argparse.ArgumentParser(
        prog="annotate_FREEC_segments.py", description="annotate segments as gain/loss/neutral"
    )
    parser.add_argument("--input_segments_file", help="input segments file from controlFREEC", required=True, type=str)
    parser.add_argument(
        "--gain_cutoff",
        help="fold-change cutoff for gain annotation",
        required=True,
        type=float,
        default=1.03,
    )
    parser.add_argument(
        "--loss_cutoff",
        help="fold-change cutoff for loss annotation",
        required=True,
        type=float,
        default=0.97,
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    df_segments = pd.read_csv(args.input_segments_file,sep='\t')
    
    gain_cutoff = args.gain_cutoff
    loss_cutoff = args.loss_cutoff
    df_segments['alteration']=df_segments['median_ratio'].apply(lambda x: 'gain' if x >= gain_cutoff else ('loss' if ((x <= loss_cutoff) & (x>-1)) else 'neutral'))

    out_annotated_file = os.path.basename(args.input_segments_file) + '_annotated.txt'
    df_segments.to_csv(out_annotated_file, sep='\t',index=False)
    out_CNVs_file = os.path.basename(args.input_segments_file) + '_CNVs.bed'
    df_segments[df_segments['alteration']!='neutral'][['chr','start','end','median_ratio']].to_csv(out_CNVs_file, sep='\t',index=False,header=None)

    logger.info("output files:")
    logger.info(out_annotated_file)
    logger.info(out_CNVs_file)


if __name__ == "__main__":
    run(sys.argv)