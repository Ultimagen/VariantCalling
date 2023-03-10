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
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse
import logging
import pickle
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam
import tqdm

from ugvc.comparison import vcf_pipeline_utils
from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.filtering import variant_filtering_utils
from ugvc.filtering.blacklist import blacklist_cg_insertions, merge_blacklists
from ugvc.vcfbed import vcftools


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap_var = argparse.ArgumentParser(prog="filter_variants_pipeline.py", description="Filter VCF")
    ap_var.add_argument("--input_file", help="Name of the input VCF file", type=str, required=True)
    ap_var.add_argument("--model_file", help="Pickle model file", type=str, required=False)
    ap_var.add_argument("--model_name", help="Model file", type=str, required=False)
    ap_var.add_argument(
        "--hpol_filter_length_dist",
        nargs=2,
        type=int,
        help="Length and distance to the hpol run to mark",
        default=[10, 10],
    )
    ap_var.add_argument("--runs_file", help="Homopolymer runs file", type=str, required=True)
    ap_var.add_argument("--blacklist", help="Blacklist file", type=str, required=False)
    ap_var.add_argument(
        "--blacklist_cg_insertions",
        help="Should CCG/GGC insertions be filtered out?",
        action="store_true",
    )
    ap_var.add_argument("--reference_file", help="Indexed reference FASTA file", type=str, required=True)
    ap_var.add_argument("--output_file", help="Output VCF file", type=str, required=True)
    ap_var.add_argument("--is_mutect", help="Is the input a result of mutect", action="store_true")
    ap_var.add_argument(
        "--flow_order",
        help="Sequencing flow order (4 cycle)",
        required=False,
        default=DEFAULT_FLOW_ORDER,
    )
    ap_var.add_argument(
        "--annotate_intervals",
        help="interval files for annotation (multiple possible)",
        required=False,
        type=str,
        default=None,
        action="append",
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
    logger.info("Reading VCF")

    try:
        df = vcftools.get_vcf_df(args.input_file)
        df, _ = vcf_pipeline_utils.annotate_concordance(
            df,
            args.reference_file,
            runfile=args.runs_file,
            flow_order=args.flow_order,
            annotate_intervals=args.annotate_intervals,
        )

        if args.is_mutect:
            df["qual"] = df["tlod"].apply(lambda x: max(x) if isinstance(x, tuple) else 50) * 10
            df["mleaf"] = df["af"]  # parameters needed for filtering
            df["mleac"] = df["ac"]  # parameters needed for filtering
            if "xc" not in df.columns:
                df["xc"] = 1

        if args.model_file is not None:
            df.loc[df["gt"] == (1, 1), "sor"] = 0.5
            with open(args.model_file, "rb") as model_file:
                models_dict = pickle.load(model_file)
            model_name = args.model_name
            models = models_dict[model_name]

            model_clsf = models

            logger.info("Applying classifier")
            df = variant_filtering_utils.add_grouping_column(
                df, variant_filtering_utils.get_training_selection_functions(), "group"
            )

        if args.blacklist is not None:
            with open(args.blacklist, "rb") as blf:
                blacklists = pickle.load(blf)
            blacklist_app = [x.apply(df) for x in blacklists]
            blacklist = merge_blacklists(blacklist_app)
        else:
            blacklist = pd.Series("PASS", index=df.index, dtype=str)

        if args.blacklist_cg_insertions:
            cg_blacklist = blacklist_cg_insertions(df)
            blacklist = merge_blacklists([cg_blacklist, blacklist])

        if args.model_file is not None:
            predictions = model_clsf.predict(df)
            predictions = np.array(predictions)

            logger.info("Applying classifier proba")
            predictions_score = model_clsf.predict(df, get_numbers=True)
            prediction_fpr = variant_filtering_utils.tree_score_to_fpr(df, predictions_score, model_clsf.tree_score_fpr)
            # Do not output FPR if it could not be calculated from the calls
            output_fpr = True
            if len(set(prediction_fpr)) == 1:
                logger.info("FPR not calculated, skipping")
                output_fpr = False

            predictions_score = np.array(predictions_score)
            group = df["group"]

        hmer_run = np.array(df.close_to_hmer_run | df.inside_hmer_run)

        logger.info("Writing")
        skipped_records = 0

        with pysam.VariantFile(args.input_file) as infile:
            hdr = infile.header

            protected_add(hdr.info, "HPOL_RUN", 1, "Flag", "In or close to homopolymer run")
            if args.model_file is not None:
                protected_add(hdr.filters, "LOW_SCORE", None, None, "Low decision tree score")
            if args.blacklist is not None:
                protected_add(hdr.info, "BLACKLST", ".", "String", "blacklist")
            if args.blacklist_cg_insertions:
                protected_add(
                    hdr.filters,
                    "CG_NON_HMER_INDEL",
                    None,
                    None,
                    "Insertion/deletion of CG",
                )

            if args.model_file is not None:
                protected_add(hdr.info, "TREE_SCORE", 1, "Float", "Filtering score")
                protected_add(hdr.info, "FPR", 1, "Float", "False Positive rate(1/MB)")
            protected_add(
                hdr.info,
                "VARIANT_TYPE",
                1,
                "String",
                "Variant type (snp, h-indel, non-h-indel)",
            )
            with pysam.VariantFile(args.output_file, mode="w", header=hdr) as outfile:
                for i, rec in tqdm.tqdm(enumerate(infile)):
                    if hmer_run[i]:
                        rec.info["HPOL_RUN"] = True
                    if args.model_file is not None:
                        if predictions[i] == "fp":
                            if "PASS" in rec.filter.keys():
                                del rec.filter["PASS"]
                            rec.filter.add("LOW_SCORE")
                        rec.info["TREE_SCORE"] = predictions_score[i]
                        if output_fpr:
                            rec.info["FPR"] = prediction_fpr[i]
                        rec.info["VARIANT_TYPE"] = group[i]
                    if args.blacklist is not None:
                        if blacklist[i] != "PASS":
                            blacklists_info = []
                            for value in blacklist[i].split(";"):
                                if value != "PASS":
                                    blacklists_info.append(value)
                            if len(blacklists_info) != 0:
                                rec.info["BLACKLST"] = blacklists_info
                    if len(rec.filter) == 0:
                        rec.filter.add("PASS")

                    # fix the alleles of form <1> that our GATK adds
                    rec.ref = rec.ref if re.match(r"<[0-9]+>", rec.ref) is None else "*"
                    rec.alleles = (y if re.match(r"<[0-9]+>", y) is None else "*" for y in rec.alleles)

                    # Removing the edge case of multiple * alleles passed due to
                    # the above correction
                    if len(rec.alleles) != len(set(rec.alleles)):
                        skipped_records += 1
                        continue
                    outfile.write(rec)

        cmd = ["bcftools", "index", "-t", args.output_file]
        subprocess.check_call(cmd)
        logger.info("Removed %s malformed records", skipped_records)
        logger.info("Variant filtering run: success")

    except Exception as err:
        exc_info = sys.exc_info()
        logger.error(*exc_info)
        logger.error("Variant filtering run: failed")
        raise err


if __name__ == "__main__":
    run(sys.argv[1:])
