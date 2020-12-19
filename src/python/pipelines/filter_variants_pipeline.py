import pathmagic
import python.pipelines.variant_filtering_utils as variant_filtering_utils
import python.modules.variant_annotation as annotation
import python.vcftools as vcftools
import argparse
import pysam
import numpy as np
import sys
import tqdm
import subprocess
import pandas as pd
import re
import dill as pickle

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
                help="Is the input a result of mutect", action="store_true", default=False)
args = ap.parse_args()

try:
    print("Reading VCF", flush=True, file=sys.stderr)
    df = vcftools.get_vcf_df(args.input_file)
    print("Adding hpol run info", flush=True, file=sys.stderr)
    min_hmer_run_length, max_distance = args.hpol_filter_length_dist
    df = annotation.close_to_hmer_run(df, args.runs_file,
                                      min_hmer_run_length=min_hmer_run_length,
                                      max_distance=max_distance)
    print("Classifying indel/SNP", flush=True, file=sys.stderr)
    df = annotation.classify_indel(df)
    print("Classifying hmer/non-hmer indel", flush=True, file=sys.stderr)
    df = annotation.is_hmer_indel(df, args.reference_file)
    print("Reading motif info", flush=True, file=sys.stderr)
    df = annotation.get_motif_around(df, 5, args.reference_file)
    df.loc[pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'

    if args.is_mutect:
        df['qual'] = df['tlod'].apply(lambda x: max(x))

    df.loc[df['gt'] == (1, 1), 'sor'] = 0.5
    models_dict = pickle.load(open(args.model_file, "rb"))
    model_name = args.model_name
    models = models_dict[model_name]

    if type(models) == list or type(models) == tuple:
        model_clsf = models[0]
        model_scor = models[1]
        is_decision_tree = True
    else:
        model_clsf = models
        is_decision_tree = False

    print("Applying classifier", flush=True, file=sys.stderr)
    df = variant_filtering_utils.add_grouping_column(df,
                                                     variant_filtering_utils.get_training_selection_functions(),
                                                     "group")

    if args.blacklist is not None:
        blacklists = pickle.load(open(args.blacklist, "rb"))
        blacklist_app = [x.apply(df) for x in blacklists]
        blacklist = variant_filtering_utils.merge_blacklists(blacklist_app)
    else:
        blacklists = []
        blacklist = pd.Series('PASS', index=df.index, dtype=str)

    if args.blacklist_cg_insertions:
        cg_blacklist = variant_filtering_utils.blacklist_cg_insertions(df)
        blacklist = variant_filtering_utils.merge_blacklists([cg_blacklist, blacklist])

    predictions = model_clsf.predict(df)
    print("Applying regressor", flush=True, file=sys.stderr)

    predictions_score = model_scor.predict(df)

    predictions = np.array(predictions)
    predictions_score = np.array(predictions_score)

    hmer_run = np.array(df.close_to_hmer_run | df.inside_hmer_run)

    print("Writing", flush=True)
    skipped_records = 0
    with pysam.VariantFile(args.input_file) as infile:
        hdr = infile.header
        hdr.info.add("HPOL_RUN", 1, "Flag", "In or close to homopolymer run")

        for b in blacklists:
            hdr.filters.add(b.annotation, None, None, b.description)

        if args.blacklist_cg_insertions:
            hdr.filters.add("CG_NON_HMER_INDEL", None, None, "Insertion/deletion of CG")

        if is_decision_tree:
            hdr.info.add("TREE_SCORE", 1, "Float", "Filtering score")
        with pysam.VariantFile(args.output_file, mode="w", header=hdr) as outfile:
            for i, rec in tqdm.tqdm(enumerate(infile)):
                pass_flag = True
                if hmer_run[i]:
                    rec.info["HPOL_RUN"] = True
                if blacklist[i] != "PASS":
                    for v in blacklist[i].split(";"):
                        if v != "PASS":
                            rec.filter.add(v)
                            pass_flag = False
                if pass_flag:
                    rec.filter.add("PASS")
                if is_decision_tree:
                    rec.info["TREE_SCORE"] = predictions_score[i]

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
    print(f"Removed {skipped_records} malformed records", file=sys.stderr, flush=True)
    print("Variant filtering run: success", file=sys.stderr, flush=True)
except Exception as err:
    exc_info = sys.exc_info()
    print(*exc_info, file=sys.stderr, flush=True)
    print("Variant filtering run: failed", file=sys.stderr, flush=True)
    raise(err)
