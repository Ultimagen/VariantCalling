import pathmagic
import logging
import argparse
import pysam
import numpy as np
import sys
import tqdm
import subprocess
import pandas as pd
import re
import pickle

import python.modules.variant_annotation as annotation
import python.pipelines.variant_filtering_utils as variant_filtering_utils
import python.vcftools as vcftools
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils

ap = argparse.ArgumentParser(
    prog="filter_variants_pipeline.py", description="Filter VCF")
ap.add_argument("--input_file", help="Name of the input VCF file",
                type=str, required=True)
ap.add_argument("--model_file", help="Pickle model file",
                type=str, required=True)
ap.add_argument("--model_name", help="Model file", type=str, required=True)
ap.add_argument("--hpol_filter_length_dist", nargs=2, type=int, help='Length and distance to the hpol run to mark',
                default=[10, 10])
ap.add_argument("--runs_file", help="Homopolymer runs file",
                type=str, required=True)
ap.add_argument("--blacklist", help="Blacklist file", type=str, required=False)
ap.add_argument("--blacklist_cg_insertions", help="Should CCG/GGC insertions be filtered out?",
                action="store_true",)
ap.add_argument("--reference_file",
                help="Indexed reference FASTA file", type=str, required=True)
ap.add_argument("--output_file", help="Output VCF file",
                type=str, required=True)
ap.add_argument("--is_mutect",
                help="Is the input a result of mutect", action="store_true")
ap.add_argument("--flow_order",
                help="Sequencing flow order (4 cycle)", required=False, default="TACG")
ap.add_argument("--annotate_intervals", help='interval files for annotation (multiple possible)', required=False,
                type=str, default=None, action='append')
args = ap.parse_args()

try:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Reading VCF")
    df = vcftools.get_vcf_df(args.input_file)

    df, annots = vcf_pipeline_utils.annotate_concordance(df, args.reference_file,
                                                         runfile=args.runs_file,
                                                         annotate_intervals=args.annotate_intervals)

    # logger.info("Adding hpol run info")
    # min_hmer_run_length, max_distance = args.hpol_filter_length_dist
    # df = annotation.close_to_hmer_run(df, args.runs_file,
    #                                   min_hmer_run_length=min_hmer_run_length,
    #                                   max_distance=max_distance)
    # logger.info("Classifying indel/SNP")
    # df = annotation.classify_indel(df)
    # logger.info("Classifying hmer/non-hmer indel")
    # df = annotation.is_hmer_indel(df, args.reference_file)
    # logger.info("Reading motif info")
    # df = annotation.get_motif_around(df, 5, args.reference_file)
    # df.loc[pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'
    # logger.info("Cycle skip info")
    # df = annotation.annotate_cycle_skip(df, flow_order=args.flow_order)

    if args.is_mutect:
        df['qual'] = df['tlod'].apply(lambda x: max(x) if type(x) == tuple else 50) * 10

    df.loc[df['gt'] == (1, 1), 'sor'] = 0.5
    with open(args.model_file, "rb") as mf:
        models_dict = pickle.load(mf)
    model_name = args.model_name
    models = models_dict[model_name]

    if type(models) == list or type(models) == tuple:
        model_clsf = models[0]
        model_scor = models[1]
        is_decision_tree = True
    else:
        model_clsf = models
        is_decision_tree = False

    logger.info("Applying classifier")
    df = variant_filtering_utils.add_grouping_column(df,
                                                     variant_filtering_utils.get_training_selection_functions(),
                                                     "group")

    # In some cases we get qual=Nan, until we will solve that, we remove such variants (we have very few of them)
    qual_na = np.isnan(df['qual'])
    df = df[~qual_na]
    if args.blacklist is not None:
        with open(args.blacklist, "rb") as blf:
            blacklists = pickle.load(blf)
        blacklist_app = [x.apply(df) for x in blacklists]
        blacklist = variant_filtering_utils.merge_blacklists(blacklist_app)
    else:
        blacklists = []
        blacklist = pd.Series('PASS', index=df.index, dtype=str)

    if args.blacklist_cg_insertions:
        cg_blacklist = variant_filtering_utils.blacklist_cg_insertions(df)
        blacklist = variant_filtering_utils.merge_blacklists([cg_blacklist, blacklist])

    predictions = model_clsf.predict(df)
    predictions = np.array(predictions)
    if is_decision_tree:
        logger.info("Applying regressor")
        predictions_score = model_scor.predict(df)
        prediction_fpr = variant_filtering_utils.tree_score_to_fpr(df, predictions_score, model_scor.tree_score_fpr)
        predictions_score = np.array(predictions_score)
        group = df['group']


    hmer_run = np.array(df.close_to_hmer_run | df.inside_hmer_run)

    logger.info("Writing")
    skipped_records = 0
    na_till_now = 0
    with pysam.VariantFile(args.input_file) as infile:
        hdr = infile.header
        hdr.info.add("HPOL_RUN", 1, "Flag", "In or close to homopolymer run")
        hdr.filters.add("LOW_SCORE", None, None, "Low decision tree score")
        for b in blacklists:
            hdr.filters.add(b.annotation, None, None, b.description)

        if args.blacklist_cg_insertions:
            hdr.filters.add("CG_NON_HMER_INDEL", None, None, "Insertion/deletion of CG")

        if is_decision_tree:
            hdr.info.add("TREE_SCORE", 1, "Float", "Filtering score")
            hdr.info.add("FPR", 1, "Float", "False Positive rate(1/MB)")
            hdr.info.add("VARIANT_TYPE", 1,  "String", "Variant type (snp, h-indel, non-h-indel)")
        qual_na_ind = pd.Series(np.where(qual_na)[0])
        with pysam.VariantFile(args.output_file, mode="w", header=hdr) as outfile:
            for i, rec in tqdm.tqdm(enumerate(infile)):
                if i in qual_na_ind:
                    na_till_now = na_till_now + 1
                    continue
                pass_flag = True
                apdated_i=i-na_till_now
                if hmer_run[apdated_i]:
                    rec.info["HPOL_RUN"] = True
                if predictions[apdated_i] == 'fp':
                    rec.filter.add("LOW_SCORE")
                    pass_flag = False
                if blacklist[apdated_i] != "PASS":
                    for v in blacklist[apdated_i].split(";"):
                        if v != "PASS":
                            rec.filter.add(v)
                            pass_flag = False
                if pass_flag:
                    rec.filter.add("PASS")
                if is_decision_tree:
                    rec.info["TREE_SCORE"] = predictions_score[apdated_i]
                    rec.info["FPR"] = prediction_fpr[apdated_i]
                    rec.info["VARIANT_TYPE"] = group[apdated_i]

                # fix the alleles of form <1> that our GATK adds
                rec.ref = rec.ref if re.match(
                    r'<[0-9]+>', rec.ref) is None else '*'
                rec.alleles = tuple(
                    [y if re.match(r'<[0-9]+>', y) is None else '*' for y in rec.alleles])

                # Removing the edge case of multiple * alleles passed due to
                # the above correction
                if len(rec.alleles) != len(set(rec.alleles)):
                    skipped_records += 1
                    continue
                outfile.write(rec)

    cmd = ['bcftools', 'index', '-t', args.output_file]
    subprocess.check_call(cmd)
    logger.info(f"Removed {skipped_records} malformed records")
    logger.info("Variant filtering run: success")
except Exception as err:
    exc_info = sys.exc_info()
    logger.error(*exc_info)
    logger.error("Variant filtering run: failed")
    raise(err)
