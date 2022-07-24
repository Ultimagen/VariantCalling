import argparse
import os
import warnings

warnings.filterwarnings("ignore")

bedtools = "/home/ubuntu/miniconda3/envs/genomics.py3/bin/bedtools"
bedmap = "/home/ubuntu/miniconda3/envs/genomics.py3/bin/bedmap"


def filter_by_lcr(bed_file, lcr_cutoff, lcr_file, prefix):
    out_lcr_file = prefix + '/' + bed_file.rstrip(".bed") + ".lcr.bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(lcr_cutoff)
        + " -a "
        + bed_file
        + " -b "
        + lcr_file
        + " > "
        + out_lcr_file
    )
    os.system(cmd)
    out_lcr_filtered_out_file = prefix + '/' + bed_file.rstrip(".bed") + ".lcr.filtered_out.bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(lcr_cutoff)
        + " -a "
        + bed_file
        + " -b "
        + out_lcr_file
        + " > "
        + out_lcr_filtered_out_file
    )
    os.system(cmd)
    out_lcr_annotate_file = prefix + '/' + bed_file.rstrip(".bed") + ".lcr.annotate.bed"
    cmd = (
        "cat "
        + out_lcr_filtered_out_file
        + ' | awk \'{print $1"\t"$2"\t"$3"\tUG-CNV-LCR"}\' > '
        + out_lcr_annotate_file
    )
    os.system(cmd)

    return out_lcr_annotate_file


def filter_by_blocklist(bed_file, diff_cutoff, diff_bed_file, prefix):
    out_blocklist_file = prefix + '/' + bed_file.rstrip(".bed") + ".blocklist.bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(diff_cutoff)
        + " -a "
        + bed_file
        + " -b "
        + diff_bed_file
        + " > "
        + out_blocklist_file
    )
    out_blocklist_filtered_out_file = prefix + '/' + bed_file.rstrip(".bed") + ".blocklist.filtered_out.bed"
    os.system(cmd)
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(diff_cutoff)
        + " -a "
        + bed_file
        + " -b "
        + out_blocklist_file
        + " > "
        + out_blocklist_filtered_out_file
    )
    os.system(cmd)
    out_blocklist_annotate_file = prefix + '/' + bed_file.rstrip(".bed") + ".blocklist.annotate.bed"
    cmd = (
        "cat "
        + out_blocklist_filtered_out_file
        + ' | awk \'{print $1"\t"$2"\t"$3"\tBLOCKLIST"}\' > '
        + out_blocklist_annotate_file
    )
    os.system(cmd)

    return out_blocklist_annotate_file


def filter_by_length(bed_file, length_cutoff, prefix):

    out_len_file = prefix + '/' + bed_file.rstrip(".bed") + ".len.bed"
    cmd = "awk '$3-$2<" + str(length_cutoff) + "' " + bed_file + " > " + out_len_file
    os.system(cmd)
    out_len_annotate_file = prefix + '/' + bed_file.rstrip(".bed") + ".len.annotate.bed"
    cmd = (
        "cat "
        + out_len_file
        + ' | awk \'{print $1"\t"$2"\t"$3"\tLEN"}\' > '
        + out_len_annotate_file
    )
    os.system(cmd)
    return out_len_annotate_file


def annotate_bed(bed_file, lcr_cutoff, lcr_file, diff_cutoff, diff_bed_file, prefix, length_cutoff=10000):
    # get filters regions
    lcr_bed_file = filter_by_lcr(bed_file, lcr_cutoff, lcr_file, prefix)
    black_bed_file = filter_by_blocklist(bed_file, diff_cutoff, diff_bed_file, prefix)
    length_bed_file = filter_by_length(bed_file, length_cutoff, prefix)

    # merge all filters and sort
    cmd = "cat " + lcr_bed_file + " " + black_bed_file + " " + length_bed_file + " > filters.annotate.unsorted.bed"
    os.system(cmd)
    cmd = bedtools + " sort -i filters.annotate.unsorted.bed > filters.annotate.bed"
    os.system(cmd)
    cmd = bedtools + " sort -i " + bed_file + " > " + bed_file.rstrip(".bed") + ".sorted.bed"
    os.system(cmd)

    # annotate bed files by filters
    cmd = (
        bedmap
        + " --echo --echo-map-id-uniq --delim '\\t' "
        + bed_file.rstrip(".bed")
        + ".sorted.bed"
        + " filters.annotate.bed"
        + " > "
        + bed_file.rstrip(".bed")
        + "unsorted.annotate.bed"
    )
    os.system(cmd)
    cmd = (
        "sort -k1,1V -k2,2n -k3,3n "
        + bed_file.rstrip(".bed")
        + "unsorted.annotate.bed > "
        + bed_file.rstrip(".bed")
        + ".annotate.bed"
    )
    os.system(cmd)
    cmd = (
        "cat " + bed_file.rstrip(".bed") + ".annotate.bed | awk '$5==\"\"' > " + bed_file.rstrip(".bed") + ".filter.bed"
    )
    os.system(cmd)

    out_annotate_file = bed_file.rstrip(".bed") + ".annotate.bed"
    out_filtered_file = bed_file.rstrip(".bed") + ".filter.bed"
    return [out_annotate_file, out_filtered_file]


def main():
    args = argparse.ArgumentParser(
        prog="filter_sample_cnvs.py", description="Filter cnvs bed file by: UG-CNV-LCR , blocklist , length"
    )
    args.add_argument("--input_bed_file", help="input bed file with .bed suffix", required=True, type=str)
    args.add_argument(
        "--intersection_cutoff",
        help="intersection cutoff for bedtools substruct function",
        required=True,
        type=float,
        default=0.5,
    )
    args.add_argument("--coverage_lcr_file", help="UG-CNV-LCR bed file", required=True, type=str)
    args.add_argument("--blocklist", help="blocklist bed file", required=True, type=str)
    args.add_argument("--min_cnv_length", required=True, type=int, default=10000)
    args.add_argument("--out_directory", help="out directory where intermediate and output files will be saved."
                                               " if not supplied all files will be written to current directory",
                      required=False, type=str)

    args = args.parse_args()

    prefix = ""
    if args.out_prefix:
        prefix = args.out_prefix
    [out_annotate_bed_file, out_filtered_bed_file] = annotate_bed(
        args.input_bed_file,
        args.intersection_cutoff,
        args.coverage_lcr_file,
        args.intersection_cutoff,
        args.blocklist,
        prefix,
        args.min_cnv_length
    )

    print("output files:")
    print(out_annotate_bed_file)
    print(out_filtered_bed_file)


if __name__ == "__main__":
    main()
