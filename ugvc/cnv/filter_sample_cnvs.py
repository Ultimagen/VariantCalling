import os
import argparse
import warnings

warnings.filterwarnings('ignore')

bedtools = '/home/ubuntu/miniconda3/envs/genomics.py3/bin/bedtools'
bedmap = '/home/ubuntu/miniconda3/envs/genomics.py3/bin/bedmap'


def filter_by_lcr(bed_file, lcr_cutoff, lcr_file):

    cmd = bedtools + ' subtract -N -f ' + str(lcr_cutoff) + ' -a ' + bed_file + ' -b ' + lcr_file + \
          ' > ' + bed_file.rstrip('.bed') + '.lcr.bed'
    os.system(cmd)
    cmd = bedtools + ' subtract -N -f ' + str(lcr_cutoff) + ' -a ' + bed_file + ' -b ' + \
          bed_file.rstrip('.bed') + '.lcr.bed' + ' > ' + bed_file.rstrip('.bed') + '.lcr_filtered_out.bed'
    os.system(cmd)
    cmd = 'cat ' + bed_file.rstrip('.bed') + '.lcr_filtered_out.bed' + \
          ' | awk \'{print $1"\t"$2"\t"$3"\tUG-CNV-LCR"}\' > ' + bed_file.rstrip('.bed') + '.lcr.annotate.bed'
    os.system(cmd)

    out_file = bed_file.rstrip('.bed') + '.lcr.annotate.bed'
    return out_file


def filter_by_blocklist(bed_file, diff_cutoff, diff_bed_file):
    cmd = bedtools + ' subtract -N -f ' + str(diff_cutoff) + ' -a ' + bed_file + ' -b ' + diff_bed_file + ' > ' \
          + bed_file.rstrip('.bed') + '.blocklist.bed'
    os.system(cmd)
    cmd = bedtools + ' subtract -N -f ' + str(diff_cutoff) + ' -a ' + bed_file + ' -b ' + \
          bed_file.rstrip('.bed') + '.blocklist.bed' + ' > ' + bed_file.rstrip('.bed') + '.blocklist.filtered_out.bed'
    os.system(cmd)
    cmd = 'cat ' +  bed_file.rstrip('.bed') + '.blocklist.filtered_out.bed' + \
          ' | awk \'{print $1"\t"$2"\t"$3"\tBLOCKLIST"}\' > ' + bed_file.rstrip('.bed') + '.blocklist.annotate.bed'
    os.system(cmd)

    out_file = bed_file.rstrip('.bed') + '.blocklist.annotate.bed'
    return out_file


def filter_by_length(bed_file, length_cutoff):

    cmd = 'awk \'$3-$2<' + str(length_cutoff) + '\' ' + bed_file + ' > ' + bed_file.rstrip('.bed') + '.len.bed'
    os.system(cmd)
    cmd = 'cat ' + bed_file.rstrip('.bed') + '.len.bed' + ' | awk \'{print $1"\t"$2"\t"$3"\tLEN"}\' > ' \
          + bed_file.rstrip('.bed') + '.len.annotate.bed'
    os.system(cmd)
    out_file = bed_file.rstrip('.bed') + '.len.annotate.bed'
    return out_file


def annotate_bed(bed_file, lcr_cutoff, lcr_file, diff_cutoff, diff_bed_file, length_cutoff=10000):
    # get filters regions
    lcr_bed_file = filter_by_lcr(bed_file, lcr_cutoff, lcr_file)
    black_bed_file = filter_by_blocklist(bed_file, diff_cutoff, diff_bed_file)
    length_bed_file = filter_by_length(bed_file, length_cutoff)

    # merge all filters and sort
    cmd = 'cat ' + lcr_bed_file + ' ' + black_bed_file + ' ' + length_bed_file + ' > filters.annotate.unsorted.bed'
    os.system(cmd)
    cmd = bedtools + ' sort -i filters.annotate.unsorted.bed > filters.annotate.bed'
    os.system(cmd)
    cmd = bedtools + ' sort -i ' + bed_file + ' > ' + bed_file.rstrip('.bed') + '.sorted.bed'
    os.system(cmd)

    # annotate bed files by filters
    cmd = bedmap + ' --echo --echo-map-id-uniq --delim \'\\t\' ' + bed_file.rstrip('.bed') + '.sorted.bed' +\
          ' filters.annotate.bed' + ' > ' + bed_file.rstrip('.bed') + '.annotate.bed'
    print(cmd)
    os.system(cmd)
    cmd = 'cat ' + bed_file.rstrip('.bed') + '.annotate.bed | awk \'$5==""\' > ' +\
          bed_file.rstrip('.bed') + '.filter.bed'
    os.system(cmd)



args = argparse.ArgumentParser(
    prog="filter_sample_cnvs.py",
    description="Filter cnvs bed file by: UG-CNV-LCR , blocklist , length"
)
args.add_argument("--sample_name", required=True, type=str)
args.add_argument("--intersection_cutoff", help="intersection cutoff for bedtools substruct function",
                  required=True, type=float , default=0.5)
args.add_argument("--coverage_lcr_file", help="UG-CNV-LCR bed file", required=True, type=str)
args.add_argument("--blocklist", help="blocklist bed file", required=True, type=str)
args.add_argument("--min_cnv_length", required=True, type=int, default=10000)

args = args.parse_args()

bed_file = args.sample_name + ".cnvs.bed"
annotate_bed(bed_file, args.intersection_cutoff,
             args.coverage_lcr_file, args.intersection_cutoff,
             args.blocklist, args.min_cnv_length)
