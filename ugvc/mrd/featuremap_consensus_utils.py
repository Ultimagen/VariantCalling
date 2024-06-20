from __future__ import annotations

from collections import defaultdict
from os.path import basename
from os.path import join as pjoin

import numpy as np
import pysam


def write_a_pileup_record(
    record_dict: dict, rec_id: str, out_fh: pysam.VariantFile, header: pysam.VariantHeader, min_qual: int
):
    info_fields = [
        "X_QUAL",
        "X_RN",
    ]

    format_fields = ["GT", "AD", "DP", "VAF"]

    rec = header.new_record()
    rec.chrom = rec_id.split("-")[0]
    rec.pos = int(rec_id.split("-")[1])
    rec.ref = rec_id.split("-")[2]
    rec.alts = [rec_id.split("-")[3]]
    rec.id = "."
    rec.qual = np.mean(record_dict["X_QUAL"])
    record_dict["X_RN"] = "|".join(record_dict["X_RN"])
    curr_filter = "PASS" if rec.qual >= min_qual else "FAIL"
    record_dict["VAF"] = record_dict["FILTERED_COUNT"] / record_dict["DP"]
    rec.filter.add(curr_filter)

    for info_field in info_fields:
        rec.info[info_field] = record_dict[info_field]

    record_dict["GT"] = (0, 1)
    record_dict["AD"] = [record_dict["FILTERED_COUNT"], record_dict["DP"] - record_dict["FILTERED_COUNT"]]
    for format_field in format_fields:
        rec.samples["SAMPLE"][format_field] = record_dict[format_field]

    out_fh.write(rec)
    return rec


def pileup_featuremap(featuremap: str, output_dir: str, genomic_interval: str, min_qual: int):
    """
    Pileup featuremap vcf to a reagular vcf, save the mean quality and read count
    """
    cons_dict = defaultdict(dict)
    # process genomic interval input
    if genomic_interval == "":
        chrom = None
        start = None
        end = None
    if genomic_interval != "":
        genomic_interval_list = genomic_interval.split(":")
        chrom = genomic_interval_list[0]
        start = int(genomic_interval_list[1].split("-")[0])
        end = int(genomic_interval_list[1].split("-")[1])

    # generate a new header
    orig_header = pysam.VariantFile(featuremap).header
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.filters.add("FAIL", None, None, "Mean quality below threshold")
    header.info.add(
        "FILTERED_COUNT",
        1,
        "Integer",
        "Number of reads containing this location that agree with alternative according to fitler",
    )
    header.info.add("X_READ_COUNT", 1, "Integer", "Number of reads containing this location")
    header.info.add("X_QUAL", ".", "Float", "Quality of reads containing this location")
    header.info.add("X_RN", 1, "String", "Read Name of reads containing this location, entries spearated by |")
    header.info.add("X_AF", 1, "Float", "Allele frequency")
    header.formats.add("GT", 1, "String", "Genotype")
    header.formats.add("DP", 1, "Integer", "Read depth")
    header.formats.add("AD", "R", "Integer", "Allelic depths")
    header.formats.add("VAF", 1, "Float", "Allele frequency")
    # copy contigs from original header
    for contig in orig_header.contigs:
        header.contigs.add(contig)
    header.add_sample("SAMPLE")

    # open an output file
    output_vcf = pjoin(
        output_dir, basename(featuremap).replace(".vcf.gz", f".{genomic_interval.replace(':', '-')}.pileup.vcf.gz")
    )
    out_fh = pysam.VariantFile(output_vcf, "w", header=header)

    cons_dict = defaultdict(dict)
    with pysam.VariantFile(featuremap) as f:
        for rec in f.fetch(chrom, start, end):
            rec_id = "-".join([str(x) for x in (rec.chrom, rec.pos, rec.ref, rec.alts[0])])
            if "FILTERED_COUNT" not in cons_dict[rec_id]:
                if len(cons_dict.keys()) > 1:
                    # write to file
                    prev_key = list(cons_dict.keys())[0]
                    write_a_pileup_record(cons_dict[prev_key], rec_id, out_fh, header, min_qual)
                    cons_dict.pop(prev_key)
                cons_dict[rec_id]["FILTERED_COUNT"] = 0
                cons_dict[rec_id]["X_QUAL"] = []
                cons_dict[rec_id]["X_RN"] = []
            cons_dict[rec_id]["FILTERED_COUNT"] += 1
            cons_dict[rec_id]["X_QUAL"] += [rec.qual]
            cons_dict[rec_id]["X_RN"] += [rec.info["X_RN"]]
            cons_dict[rec_id]["DP"] = rec.info["X_READ_COUNT"]

    out_fh.close()
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf
