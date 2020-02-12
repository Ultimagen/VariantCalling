import python.variant_filtering_utils as variant_filtering_utils
import python.vcftools as vcftools
import argparse
import pandas as pd
import pickle
import pysam
import numpy as np

ap = argparse.ArgumentParser("Filter VCF")
ap.add_argument("--input_file", help="Name of the input VCF file", type=str, required=True)
ap.add_argument("--model_file", help="Pickle model file", type=str, required=True)
ap.add_argument("--model_name", help="Model file", type=str, required=True)
ap.add_argument("--runs_file", help="Homopolymer runs file", type=str, required=True)
ap.add_argument("--reference_file", help="Indexed reference FASTA file", type=str, required=True)
ap.add_argument("--output_file", help="Output VCF file",
                type=str, required=True)
args = ap.parse_args()
df = vcftools.get_vcf_df(args.input_file)
df = vcftools.close_to_hmer_run(df, args.runs_file, min_hmer_run_length=10, max_distance=10)
df = vcftools.classify_indel(df)
df = vcftools.is_hmer_indel(df, args.reference_file)
df = vcftools.get_motif_around(df, 5, args.reference_file)

df.to_hdf(args.output_file + ".in.vcf.h5")
models_dict = pickle.load(open(args.model_file, "rb"))
models = models_dict[args.model_name]
if type(models) == list or type(models) == tuple:
    model = models[0]
else:
    model = models
predictions = model.predict(
    variant_filtering_utils.add_grouping_column(df,
                                                variant_filtering_utils.get_training_selection_functions(), "group"))

predictions = np.array(predictions)

hmer_run = np.array(df.close_to_hmer_run | df.inside_hmer_run)

with pysam.VariantFile(args.input_file) as infile:
    hdr = infile.header
    hdr.filters.add("HPOL_RUN", None, None, "Homopolymer run")
    hdr.filters.add("LOW_SCORE", None, None, "Low decision tree score")
    with pysam.VariantFile(args.output_file, mode="w", header=hdr) as outfile:
        for i, rec in enumerate(infile):
            pass_flag = True
            if hmer_run[i]:
                rec.filter.add("HPOL_RUN")
                pass_flag = False
            if predictions[i] == 'fp':
                rec.filter.add("LOW_SCORE")
                pass_flag = False
            if pass_flag:
                rec.filter.add("PASS")
            outfile.write(rec)
