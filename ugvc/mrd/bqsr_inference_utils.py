from __future__ import annotations

import os
import subprocess
from os.path import join as pjoin

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os.path import basename, dirname, isfile, join as pjoin
import subprocess
import joblib
import joblib
from joblib import dump, load, Parallel, delayed
import random
from tqdm import tqdm

from ugvc.dna.utils import revcomp
from ugvc.utils.misc_utils import set_pyplot_defaults
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe, get_motif_around


def iterate_vcf_in_chunks(vcf_file, chunk_size, model, output_file, chrstr, params, show_progress=True):
    print(f"fetching chr {chrstr} from {vcf_file}\n")
    with pysam.VariantFile(vcf_file) as vcf_in, pysam.VariantFile(output_file, "w", header=vcf_in.header) as vcf_out:

        vcf_in.header.add_meta(
            "INFO", items=[("ID", "ML_QUAL"), ("Number", 1), ("Type", "Float"), ("Description", "xgboost model score")]
        )

        # Add a comment to the output file indicating the use of an ML model
        vcf_out.header.add_line(
            '##INFO=<ID=ML_QUAL,Number=1,Type=Float,Description="Quality assigned using an ML model">'
        )

        # Calculate the total number of chunks
        # total_variants = vcf_in.count_records()
        # total_chunks = (total_variants - 1) // chunk_size + 1

        # Iterate over the variants in chunks
        variants = vcf_in.fetch(chrstr)
        progress_tracker = tqdm(total=None, disable=not show_progress)
        chunk = [None] * chunk_size        
        index = 0
        for variant in variants:
            chunk[index] = variant
            index += 1

            if index == chunk_size:
                process_chunk(chunk, model, vcf_out, params)
                chunk = [None] * chunk_size
                index = 0
                progress_tracker.update(1)

        # Process the remaining variants (if any) in the last chunk
        if index > 0:
            process_chunk(chunk[:index], model, vcf_out, params)
            progress_tracker.update(1)

        progress_tracker.close()


def process_chunk(chunk, model, vcf_out, params):

    # Convert variants to a pandas DataFrame
    data = []
    for variant in chunk:
        d_rec = {
            "chrom": variant.chrom,
            "pos": variant.pos,
            "ref": variant.ref,
            "alt": variant.alts[0],
            **dict(variant.info),
        }
        data.append(d_rec)
    df = pd.DataFrame(data)

    # Apply the provided model to assign a new quality value
    balanced_epcr_cols = ["a_start", "ts", "ae", "te", "s2", "a3"]
    is_mixed_query = (
        "a_start >= 2 and a_start <= 4 and ts >= 2 and ts <= 4 and ae >= 2 and ae <= 4 and te >= 2 and te <= 4"
    )
    df = (
        df.rename(columns={"as": "a_start"})
        .fillna({x: 0 for x in balanced_epcr_cols})
        .astype({x: int for x in balanced_epcr_cols})
        .astype({"rq": float})
    )
    df = df.assign(
        is_reverse=(df["X_FLAGS"] & 16).astype(bool),
        qc_valid=~(df["X_FLAGS"] & 512).astype(bool),
        read_end_reached=~df["s2"].isnull(),
        strand_ratio_5=df["ts"] / (df["ts"] + df["a_start"]),
        strand_ratio_3=df["te"] / (df["te"] + df["ae"]),
    )

    df = df.assign(is_mixed=df.eval(is_mixed_query).values)

    # TODO: make this code match the featermap_to_dataframe_local
    # extract motifs and determine cskp (usually done in featuremap_to_dataframe)
    df = (
        get_motif_around(df.assign(indel=False), motif_size=4, fasta=params["ref_fasta"])
        .drop(columns=["indel"])
        .astype({"left_motif": str, "right_motif": str})
    )
    left_motif_reverse = df["left_motif"].apply(revcomp)
    right_motif_reverse = df["right_motif"].apply(revcomp)
    is_reverse = (df["X_FLAGS"] & 16).astype(bool)

    # Reverse complement the ref and alt alleles if necessary to match the read direction
    for column in ("ref", "alt"):
        df[column] = df[column].where(is_reverse, df[column].apply(revcomp))
    df.loc[:, "left_motif"] = df["left_motif"].where(is_reverse, right_motif_reverse)
    df.loc[:, "right_motif"] = df["right_motif"].where(is_reverse, left_motif_reverse)

    df.loc[:, "ref_motif"] = df["left_motif"].str.slice(-1) + df["ref"] + df["right_motif"].str.slice(0, 1)
    df.loc[:, "alt_motif"] = df["left_motif"].str.slice(-1) + df["alt"] + df["right_motif"].str.slice(0, 1)
    flow_order = "TGCA"
    df_cskp = get_cycle_skip_dataframe(flow_order=flow_order)
    df = df.set_index(["ref_motif", "alt_motif"]).join(df_cskp).reset_index()

    df = df.assign(is_cycle_skip=df["cycle_skip_status"] == "cycle-skip")

    df.loc[:, "ref_alt_bases"] = df["ref"].str.cat(df["alt"], sep="")
    df.loc[:, "ref_alt_motif"] = df["ref_motif"].str.cat(df["alt"], sep="")
    df.loc[:, "left_motif"] = df["left_motif"].str.slice(start=0, stop=3)
    df.loc[:, "right_motif"] = df["right_motif"].str.slice(start=1, stop=4)

    features = df[model.feature_names_in_]
    motif_only_features = ["ref_alt_motif", "left_motif", "right_motif"]
    categorical_features = ["cycle_skip_status"] + motif_only_features

    features = features.astype({col: "category" for col in categorical_features})

    predicted_qualities = model.predict_proba(features)

    # Assign the new quality value to each variant and write to the output file
    for i, variant in enumerate(chunk):
        # Add a new INFO field to the variant indicating the ML quality value
        variant.info["ML_QUAL"] = -10 * np.log10(predicted_qualities[i][0])
        # Write the variant with the new quality to the output VCF file
        vcf_out.write(variant)


def bqsr_inference(
    featuremap_path,
    params_path,
    model_path,
    out_path: str,
    out_basename: str = "",
    reference_fasta: str = None,
):

    set_pyplot_defaults()

    # init    
    os.makedirs(out_path, exist_ok=True)
    if len(out_basename) > 0 and not out_basename.endswith("."):
        out_basename += "."

    params = {}
    params["workdir"] = out_path
    params["out_basename"] = out_basename
    params["featuremap_path"] = featuremap_path
    params["params_path"] = params_path
    params["model_path"] = model_path
    params["ref_fasta"] = reference_fasta
    params["chunk_size"] = 20000  # vcf is loaded in chunks, this number is the amount of reads in chunk, TODO: external parameter

    def parallel_iterate_over_chunks(chrstr):
        # global params
        vcf_file = params["featuremap_path"]        
        chunk_size = params["chunk_size"]  # Number of variants in each chunk
        model = joblib.load(params["model_path"])
        vcf_out = pjoin(params["workdir"], params["out_basename"] + f"_output_{chrstr}.vcf.gz")

        iterate_vcf_in_chunks(vcf_file, chunk_size, model, vcf_out, chrstr, params)

    # inference on input featuremap
    chrlist = []
    for c in np.arange(1, 23).astype(str):
        chrlist.append("chr" + c)

    # Parallel loop on chromosomes and then unite results
    Parallel(n_jobs=-1)(delayed(parallel_iterate_over_chunks)(chr_name) for chr_name in chrlist)
    united_vcf_file = pjoin(params["workdir"], f"{params['out_basename']}.featuremap.ML_qual.vcf.gz")
    commandstr = f"bcftools concat {params['workdir']}/*featuremap_output_chr*.vcf.gz -o f{united_vcf_file} -O z"
    print(commandstr)
    subprocess.call(commandstr, shell=True)
    commandstr = f"bcftools index -t {united_vcf_file}"
    subprocess.call(commandstr, shell=True)

    return True
