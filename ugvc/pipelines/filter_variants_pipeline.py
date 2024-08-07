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
#    Filter raw GATK callset using ML model

from __future__ import annotations

import argparse
import logging
import os.path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam
import tqdm

from ugvc.filtering import variant_filtering_utils
from ugvc.filtering.blacklist import blacklist_cg_insertions, merge_blacklists
from ugvc.utils import math_utils
from ugvc.vcfbed import vcftools


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(prog="filter_variants_pipeline.py", description="Filter VCF")
    ap_var.add_argument(
        "--input_file", help="Name of the input VCF file (requires .tbi index)", type=str, required=True
    )
    ap_var.add_argument("--model_file", help="Pickle model file", type=str, required=False)
    ap_var.add_argument("--blacklist", help="Blacklist file", type=str, required=False)
    ap_var.add_argument(
        "--custom_annotations",
        help="Custom INFO annotations to read from the VCF (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
    )

    ap_var.add_argument(
        "--blacklist_cg_insertions",
        help="Should CCG/GGC insertions be filtered out?",
        action="store_true",
    )
    ap_var.add_argument("--output_file", help="Output VCF file", type=str, required=True)
    ap_var.add_argument(
        "--limit_to_contigs", help="Limit filtering to these contigs", nargs="+", type=str, default=None
    )
    return ap_var.parse_args(argv)


def protected_add(hdr, field, n_vals, param_type, description):
    if field not in hdr:
        hdr.add(field, n_vals, param_type, description)


def run(argv: list[str]):
    "POST-GATK variant filtering"
    args = parse_args(argv)
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        if args.model_file is not None:
            logger.info(f"Loading model from {args.model_file}")
            with open(args.model_file, "rb") as model_file:
                mf = pickle.load(model_file)
                model = mf["xgb"]
                transformer = mf["transformer"]
        if args.blacklist is not None:
            logger.info(f"Loading blacklist from {args.blacklist}")
            with open(args.blacklist, "rb") as blf:
                blacklists = pickle.load(blf)
        assert os.path.exists(args.input_file), f"Input file {args.input_file} does not exist"
        assert os.path.exists(args.input_file + ".tbi"), f"Index file {args.input_file}.tbi does not exist"
        with pysam.VariantFile(args.input_file) as infile:
            hdr = infile.header
            if args.model_file is not None:
                protected_add(hdr.filters, "LOW_SCORE", None, None, "Low decision tree score")
            if args.blacklist is not None or args.blacklist_cg_insertions:
                protected_add(hdr.info, "BLACKLST", ".", "String", "blacklist")
            if args.model_file is not None:
                protected_add(hdr.info, "TREE_SCORE", 1, "Float", "Filtering score")

            with pysam.VariantFile(args.output_file, mode="w", header=hdr) as outfile:
                if args.limit_to_contigs is None:
                    it = ((x, infile.fetch(str(x))) for x in infile.header.contigs.keys())
                else:
                    it = ((x, infile.fetch(x)) for x in args.limit_to_contigs)
                for contig, chunk in it:  # pylint: disable=too-many-nested-blocks
                    logger.info(f"Filtering variants from {contig}")
                    df = vcftools.get_vcf_df(
                        args.input_file, chromosome=str(contig), custom_info_fields=args.custom_annotations
                    )
                    if df.shape[0] == 0:
                        logger.info(f"No variants found on {contig}")
                        continue
                    logger.info(f"{df.shape[0]} variants found on {contig}")
                    if args.blacklist is not None:
                        blacklist_app = [x.apply(df) for x in blacklists]
                        blacklist = merge_blacklists(blacklist_app)
                        logger.info("Applying blacklist")
                    else:
                        blacklist = pd.Series("PASS", index=df.index, dtype=str)

                    if args.blacklist_cg_insertions:
                        cg_blacklist = blacklist_cg_insertions(df)
                        blacklist = merge_blacklists([cg_blacklist, blacklist])
                        logger.info("Marking CG insertions")

                    if args.model_file is not None:
                        logger.info("Applying classifier")
                        predictions, scores = variant_filtering_utils.apply_model(df, model, transformer)
                        phred_pls = math_utils.phred(scores)
                        quals = -phred_pls[:, 1:].max(axis=1) + phred_pls[:, 0]
                        quals = np.clip(quals + 30, 0, 100)

                    logger.info("Writing records")
                    for i, rec in tqdm.tqdm(enumerate(chunk)):
                        if args.model_file is not None:
                            if predictions[i] == 0:
                                if "PASS" in rec.filter.keys():
                                    del rec.filter["PASS"]
                                rec.filter.add("LOW_SCORE")
                            rec.info["TREE_SCORE"] = quals[i]
                        if blacklist is not None:
                            if blacklist[i] != "PASS":
                                blacklists_info = []
                                for value in blacklist[i].split(";"):
                                    if value != "PASS":
                                        blacklists_info.append(value)
                                if len(blacklists_info) != 0:
                                    rec.info["BLACKLST"] = blacklists_info
                        if len(rec.filter) == 0:
                            rec.filter.add("PASS")

                        outfile.write(rec)
                    logger.info(f"{contig} done")

        cmd = ["bcftools", "index", "-t", args.output_file]
        subprocess.check_call(cmd)
        logger.info("Variant filtering run: success")

    except Exception as err:
        exc_info = sys.exc_info()
        logger.error(exc_info[:2])
        logger.exception(err)
        logger.error("Variant filtering run: failed")
        raise err


if __name__ == "__main__":
    run(sys.argv[1:])
