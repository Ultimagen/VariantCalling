import os
import warnings

warnings.filterwarnings("ignore")

bedtools = "bedtools"


def filter_by_bed_file(in_bed_file, filtration_cutoff, filtering_bed_file, prefix, tag):
    out_filename = os.path.basename(in_bed_file)
    out_filtered_bed_file = prefix + out_filename.rstrip(".bed") + "." + tag + ".bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(filtration_cutoff)
        + " -a "
        + in_bed_file
        + " -b "
        + filtering_bed_file
        + " > "
        + out_filtered_bed_file
    )

    os.system(cmd)
    filtered_out_records = prefix + out_filename.rstrip(".bed") + "." + tag + ".filtered_out.bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(filtration_cutoff)
        + " -a "
        + in_bed_file
        + " -b "
        + out_filtered_bed_file
        + " > "
        + filtered_out_records
    )

    os.system(cmd)
    out_annotate_file = prefix + out_filename.rstrip(".bed") + "." + tag + ".annotate.bed"
    cmd = "cat " + filtered_out_records + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|' + tag + "\"}' > " + out_annotate_file
    os.system(cmd)

    return out_annotate_file


def filter_by_length(bed_file, length_cutoff, prefix):
    out_filename = os.path.basename(bed_file)
    out_len_file = prefix + out_filename.rstrip(".bed") + ".len.bed"
    cmd = "awk '$3-$2<" + str(length_cutoff) + "' " + bed_file + " > " + out_len_file
    os.system(cmd)
    out_len_annotate_file = prefix + out_filename.rstrip(".bed") + ".len.annotate.bed"
    cmd = "cat " + out_len_file + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|LEN"}\' > ' + out_len_annotate_file
    os.system(cmd)

    return out_len_annotate_file
