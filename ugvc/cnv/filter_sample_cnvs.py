import argparse
import logging
import os
import sys
import warnings

from ugvc import logger
from ugvc.vcfbed import filter_bed

warnings.filterwarnings("ignore")

bedtools = "bedtools"
bedmap = "bedmap"


def annotate_bed(bed_file, lcr_cutoff, lcr_file, prefix, length_cutoff=10000):
    # get filters regions
    lcr_bed_file = filter_bed.filter_by_bed_file(bed_file, lcr_cutoff, lcr_file, prefix, "UG-CNV-LCR")
    length_bed_file = filter_bed.filter_by_length(bed_file, length_cutoff, prefix)

    # merge all filters and sort
    out_filters_unsorted = prefix + "filters.annotate.unsorted.bed"
    cmd = "cat " + lcr_bed_file + " " + length_bed_file + " > " + out_filters_unsorted

    os.system(cmd)
    out_filters_sorted = prefix + "filters.annotate.bed"

    cmd = bedtools + " sort -i " + out_filters_unsorted + " > " + out_filters_sorted

    os.system(cmd)
    out_bed_file_sorted = prefix + os.path.basename(bed_file).rstrip(".bed") + ".sorted.bed"
    cmd = bedtools + " sort -i " + bed_file + " > " + out_bed_file_sorted

    os.system(cmd)

    # annotate bed files by filters
    out_unsorted_annotate = prefix + os.path.basename(bed_file).rstrip(".bed") + ".unsorted.annotate.bed"
    cmd = (
        bedmap
        + " --echo --echo-map-id-uniq --delim '\\t' "
        + out_bed_file_sorted
        + " "
        + out_filters_sorted
        + " > "
        + out_unsorted_annotate
    )
    os.system(cmd)
    # combine to 4th column

    out_combined_info = prefix + os.path.basename(bed_file).rstrip(".bed") + ".unsorted.annotate.combined.bed"
    cmd = (
        "cat "
        + out_unsorted_annotate
        + ' | awk \'{if($5=="")'
        + "{print $0}"
        + 'else{print $1"\t"$2"\t"$3"\t"$5}}\' > '
        + out_combined_info
    )
    print(cmd)
    os.system(cmd)

    out_annotate = prefix + os.path.basename(bed_file).rstrip(".bed") + ".annotate.bed"
    cmd = "sort -k1,1V -k2,2n -k3,3n " + out_combined_info + " > " + out_annotate
    os.system(cmd)

    out_filtered = prefix + os.path.basename(bed_file).rstrip(".bed") + ".filter.bed"
    cmd = "cat " + out_annotate + ' | grep -v "|" > ' + out_filtered

    os.system(cmd)

    return [out_annotate, out_filtered]


def check_path(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        logger.info("creating out directory : %s", path)


def run(argv):
    """
    Given a bed file, this script will filter it by :
    1. lcr bed (ug_cnv_lcr) file
    3. length
    output consists of 2 files:
    - annotated bed file with filtering tags
    - filtered bed file
    """
    parser = argparse.ArgumentParser(
        prog="filter_sample_cnvs.py", description="Filter cnvs bed file by: ug_cnv_lcr, length"
    )
    parser.add_argument("--input_bed_file", help="input bed file with .bed suffix", required=True, type=str)
    parser.add_argument(
        "--intersection_cutoff",
        help="intersection cutoff for bedtools substruct function",
        required=True,
        type=float,
        default=0.5,
    )
    parser.add_argument("--coverage_lcr_file", help="UG-CNV-LCR bed file", required=True, type=str)
    parser.add_argument("--min_cnv_length", required=True, type=int, default=10000)
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )

    args = parser.parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))

    prefix = ""
    if args.out_directory:
        prefix = args.out_directory
        prefix = prefix.rstrip("/") + "/"
    [out_annotate_bed_file, out_filtered_bed_file] = annotate_bed(
        args.input_bed_file, args.intersection_cutoff, args.coverage_lcr_file, prefix, args.min_cnv_length
    )

    logger.info("output files:")
    logger.info(out_annotate_bed_file)
    logger.info(out_filtered_bed_file)


if __name__ == "__main__":
    run(sys.argv)
