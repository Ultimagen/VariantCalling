from __future__ import annotations

import os
import subprocess
import tempfile
from collections import defaultdict
from os.path import basename
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pysam

from ugvc import logger


def generate_featuremap_pileup(featuremap: str, output_dir: str, min_qual: int, vcf_header: str):
    """
    Pileup featuremap vcf to a reagular vcf, save the mean quality and read count
    """
    cons_dict = defaultdict(dict)

    with pysam.VariantFile(featuremap) as f:
        for rec in f.fetch():
            rec_id = "-".join([str(x) for x in (rec.chrom, rec.pos, rec.ref, rec.alts[0])])
            cons_dict[rec_id]["X_READ_COUNT"] = rec.info["X_READ_COUNT"]
            if "FILTERED_COUNT" not in cons_dict[rec_id]:
                cons_dict[rec_id]["FILTERED_COUNT"] = 0
                cons_dict[rec_id]["X_QUAL"] = []
                cons_dict[rec_id]["X_RN"] = []
            cons_dict[rec_id]["FILTERED_COUNT"] += 1
            cons_dict[rec_id]["X_QUAL"] += [rec.qual]
            cons_dict[rec_id]["X_RN"] += [rec.info["X_RN"]]

    df_af = pd.DataFrame(cons_dict).T
    df_af["X_AF"] = df_af["FILTERED_COUNT"] / df_af["X_READ_COUNT"]
    # Save mean quality
    df_af["MEAN_QUAL"] = df_af["X_QUAL"].apply(np.mean)
    # format as a vcf file
    df_af.reset_index(inplace=True)
    df_af["CHROM"] = df_af["index"].apply(lambda x: x.split("-")[0])
    df_af["POS"] = df_af["index"].apply(lambda x: int(x.split("-")[1]))
    df_af["REF"] = df_af["index"].apply(lambda x: x.split("-")[2])
    df_af["ALT"] = df_af["index"].apply(lambda x: x.split("-")[3])
    df_af["FILTER"] = df_af["MEAN_QUAL"].apply(lambda x: "PASS" if x >= min_qual else "FAIL")
    columns_to_agg = [
        "FILTERED_COUNT",
        "X_READ_COUNT",
        "X_QUAL",
        "X_RN",
        "X_AF",
    ]
    ";".join(df_af[columns_to_agg].apply(lambda x: f"{x.name}={x.values}", axis=0))
    df_af["INFO"] = ""
    for i, row in df_af.iterrows():
        df_af.loc[i, "INFO"] = ";".join([f"{k}={v}" for k, v in row[columns_to_agg].items()])
    df_af["ID"] = "."
    output_vcf = pjoin(output_dir, basename(featuremap).replace(".vcf.gz", ".pileup.vcf"))
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        output_tmp = os.path.join(temp_dir, "featuremap.vcf")
        df_af[["CHROM", "POS", "ID", "REF", "ALT", "MEAN_QUAL", "FILTER", "INFO"]].to_csv(
            output_tmp, sep="\t", index=False, header=False, float_format="%.2f"
        )
        # add the header
        cmd = f"cat {vcf_header} {output_tmp} > {output_vcf}"
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)
        # gzip and index with bcftools
        cmd = (
            f"bcftools view {output_vcf} -Oz -o {output_vcf}.gz && bcftools index -t {output_vcf}.gz && rm {output_vcf}"
        )
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)
    return output_vcf + ".gz"
