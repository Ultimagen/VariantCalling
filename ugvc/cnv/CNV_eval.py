import argparse
import os
import sys
import pandas as pd

def intersection(sample1_basename, sample2_basename, sample1_bed_file, sample2_bed_file, intersection_cutoff=0.5):
    out_sample1_specific = f"{sample1_basename}.specific_cnv.bed"
    out_sample2_specific = f"{sample2_basename}.specific_cnv.bed"
    out_common_sample1 = f"{sample1_basename}.common_cnv.bed"
    out_common_sample2 = f"{sample2_basename}.common_cnv.bed"

    cmd = f"bedtools subtract -N -f {str(intersection_cutoff)} -a {sample1_bed_file} -b {sample2_bed_file} > {out_sample1_specific}"
    os.system(cmd)
    cmd = f"bedtools subtract -N -f {str(intersection_cutoff)} -a {sample2_bed_file} -b {sample1_bed_file} > {out_sample2_specific}"
    os.system(cmd)
    cmd = f"bedtools subtract -A -a {sample1_bed_file} -b {out_sample1_specific} > {out_common_sample1}"
    os.system(cmd)
    cmd = f"bedtools subtract -A -a {sample2_bed_file} -b {out_sample2_specific} > {out_common_sample2}"
    os.system(cmd)

    count_sample1_specific = 0
    count_sample2_specific = 0
    count_sample_1_common = 0
    count_sample_2_common = 0

    if os.stat(out_sample1_specific).st_size > 0:
        count_sample1_specific = sum(1 for line in open(out_sample1_specific))
    if os.stat(out_sample2_specific).st_size > 0:
        count_sample2_specific = sum(1 for line in open(out_sample2_specific))
    if os.stat(out_common_sample1).st_size > 0:
        count_sample_1_common = sum(1 for line in open(out_common_sample1))
    if os.stat(out_common_sample2).st_size > 0:
        count_sample_2_common = sum(1 for line in open(out_common_sample2))

    return [count_sample1_specific, count_sample2_specific, count_sample_1_common, count_sample_2_common]


def filter_len(sample1_basename, sample2_basename, len_cutoff=10000):
    sample1_specific = f"{sample1_basename}.specific_cnv.bed"
    sample2_specific = f"{sample2_basename}.specific_cnv.bed"
    common_sample1 = f"{sample1_basename}.common_cnv.bed"
    common_sample2 = f"{sample2_basename}.common_cnv.bed"

    out_sample1_specific_len = f"{sample1_basename}.specific_cnv.len.bed"
    out_sample2_specific_len = f"{sample2_basename}.specific_cnv.len.bed"
    out_sample1_common_len = f"{sample1_basename}.common_cnv.len.bed"
    out_sample2_common_len = f"{sample2_basename}.common_cnv.len.bed"

    # filter by length
    cmd = "awk '$3-$2>=10000' " + sample1_specific + " > " + out_sample1_specific_len
    os.system(cmd)
    cmd = "awk '$3-$2>=10000' " + sample2_specific + " > " + out_sample2_specific_len
    os.system(cmd)
    cmd = "awk '$3-$2>=10000' " + common_sample1 + " > " + out_sample1_common_len
    os.system(cmd)
    cmd = "awk '$3-$2>=10000' " + common_sample2 + " > " + out_sample2_common_len
    os.system(cmd)

    count_sample1_specific_len = 0
    count_sample2_specific_len = 0
    count_sample1_common_len = 0
    count_sample2_common_len = 0
    if os.stat(out_sample1_specific_len).st_size > 0:
        count_sample1_specific_len = sum(1 for line in open(out_sample1_specific_len))
    if os.stat(out_sample2_specific_len).st_size > 0:
        count_sample2_specific_len = sum(1 for line in open(out_sample2_specific_len))
    if os.stat(out_sample1_common_len).st_size > 0:
        count_sample1_common_len = sum(1 for line in open(out_sample1_common_len))
    if os.stat(out_sample2_common_len).st_size > 0:
        count_sample2_common_len = sum(1 for line in open(out_sample2_common_len))

    return [count_sample1_specific_len, count_sample2_specific_len, count_sample1_common_len, count_sample2_common_len]


def filter_cnv_lcr(sample1_basename, sample2_basename, lcr_file, intersection_cutoff=0.5):
    sample1_specific_len = f"{sample1_basename}.specific_cnv.len.bed"
    sample2_specific_len = f"{sample2_basename}.specific_cnv.len.bed"
    sample1_common_len = f"{sample1_basename}.common_cnv.len.bed"
    sample2_common_len = f"{sample2_basename}.common_cnv.len.bed"

    out_sample1_specific_len_lcr = f"{sample1_basename}.specific_cnv.len.lcr.bed"
    out_sample2_specific_len_lcr = f"{sample2_basename}.specific_cnv.len.lcr.bed"
    out_sample1_common_len_lcr = f"{sample1_basename}.common_cnv.len.lcr.bed"
    out_sample2_common_len_lcr = f"{sample2_basename}.common_cnv.len.lcr.bed"

    cmd = f"bedtools subtract -N -f {str(intersection_cutoff)} -a {sample1_specific_len} -b {lcr_file} > {out_sample1_specific_len_lcr}"
    os.system(cmd)
    cmd = f"bedtools subtract -N -f {str(intersection_cutoff)} -a {sample2_specific_len} -b {lcr_file} > {out_sample2_specific_len_lcr}"
    os.system(cmd)
    cmd = f"bedtools subtract -N -f {str(intersection_cutoff)} -a {sample1_common_len} -b {lcr_file} > {out_sample1_common_len_lcr}"
    os.system(cmd)
    cmd = f"bedtools subtract -N -f {str(intersection_cutoff)} -a {sample2_common_len} -b {lcr_file} > {out_sample2_common_len_lcr}"
    os.system(cmd)

    count_sample1_specific_len_lcr = 0
    count_sample2_specific_len_lcr = 0
    count_sample1_common_len_lcr = 0
    count_sample2_common_len_lcr = 0
    if os.stat(out_sample1_specific_len_lcr).st_size > 0:
        count_sample1_specific_len_lcr = sum(1 for line in open(out_sample1_specific_len_lcr))
    if os.stat(out_sample2_specific_len_lcr).st_size > 0:
        count_sample2_specific_len_lcr = sum(1 for line in open(out_sample2_specific_len_lcr))
    if os.stat(out_sample1_common_len_lcr).st_size > 0:
        count_sample1_common_len_lcr = sum(1 for line in open(out_sample1_common_len_lcr))
    if os.stat(out_sample2_common_len_lcr).st_size > 0:
        count_sample2_common_len_lcr = sum(1 for line in open(out_sample2_common_len_lcr))

    return [
        count_sample1_specific_len_lcr,
        count_sample2_specific_len_lcr,
        count_sample1_common_len_lcr,
        count_sample2_common_len_lcr,
    ]


def run(argv):
    """
    Runs concocrdance for 2 given CNV bed files.
    maked the following filtering:
    1. intersection : sample1 vs sample2
    2. lcr bed (ug_cnv_lcr) file
    3. length
    output directory consists of concordance summary: <sample1>_<sample2>.concordance.csv
    """
    parser = argparse.ArgumentParser(prog="CNV_eval", description="Runs concocrdance for 2 given CNV bed files.")
    parser.add_argument("--sample1_bed_file", help="sample1_cnv_bed_file", required=True, type=str)
    parser.add_argument("--sample2_bed_file", help="sample2_cnv_bed_file", required=True, type=str)
    parser.add_argument("--sample1_basename", help="sample1_basename", required=True, type=str)
    parser.add_argument("--sample2_basename", help="sample2_basename", required=True, type=str)
    parser.add_argument("--out_dir", help="output_directory", required=True, type=str)
    parser.add_argument("--ug_cnv_lcr", help="ug_cnv_lcr bed file", required=True, type=str)

    args = parser.parse_args(argv[1:])
    sample1_bed_file = args.sample1_bed_file
    sample2_bed_file = args.sample2_bed_file
    sample1_basename = args.sample1_basename
    sample2_basename = args.sample2_basename
    out_dir = args.out_dir
    ug_cnv_lcr = args.ug_cnv_lcr

    # define out dir
    out_run_dir = os.path.join(out_dir, sample1_basename + "_" + sample2_basename)
    if not os.path.isdir(out_run_dir):
        cmd = "mkdir " + out_run_dir
        os.system(cmd)
    else:
        print(f"using existing directory: {out_dir}")

    os.chdir(out_run_dir)
    d = {"sample1": [sample1_basename], "sample2": [sample2_basename]}
    df_concordance = pd.DataFrame(data=d)

    [
        df_concordance["sample1_specific"],
        df_concordance["sample2_specific"],
        df_concordance["sample1_common"],
        df_concordance["sample2_common"],
    ] = intersection(sample1_basename, sample2_basename, sample1_bed_file, sample2_bed_file)
    [
        df_concordance["sample1_specific_10K"],
        df_concordance["sample2_specific_10K"],
        df_concordance["sample1_common_10K"],
        df_concordance["sample2_common_10K"],
    ] = filter_len(sample1_basename, sample2_basename)
    [
        df_concordance["sample1_specific_10K_lcr"],
        df_concordance["sample2_specific_10K_lcr"],
        df_concordance["sample1_common_10K_lcr"],
        df_concordance["sample2_common_10K_lcr"],
    ] = filter_cnv_lcr(sample1_basename, sample2_basename, ug_cnv_lcr)

    df_concordance["percision"] = df_concordance["sample1_common_10K_lcr"] / (
        df_concordance["sample1_common_10K_lcr"] + df_concordance["sample1_specific_10K_lcr"]
    )
    df_concordance["percision"] = pd.to_numeric(df_concordance["percision"], downcast="float")
    df_concordance["recall"] = df_concordance["sample2_common_10K_lcr"] / (
        df_concordance["sample2_common_10K_lcr"] + df_concordance["sample2_specific_10K_lcr"]
    )
    df_concordance["recall"] = pd.to_numeric(df_concordance["recall"], downcast="float")

    out_concordance_file = sample1_basename + "_" + sample2_basename + ".concordance.csv"
    df_concordance.to_csv(out_concordance_file, index=False)


if __name__ == "__main__":
    run(sys.argv)
