from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum

import numpy as np
import pysam

from ugvc.dna.format import AD, DP, GT, VAF
from ugvc.mrd.featuremap_utils import FeatureMapFields

fields_to_collect = {
    "numeric_array_fields": [
        FeatureMapFields.X_SCORE.value,
        FeatureMapFields.X_EDIST.value,
        FeatureMapFields.X_LENGTH.value,
        FeatureMapFields.X_MAPQ.value,
        FeatureMapFields.X_INDEX.value,
        FeatureMapFields.X_FC1.value,
        FeatureMapFields.X_FC2.value,
        FeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
        FeatureMapFields.X_FLAGS.value,
    ],
    "string_list_fields": [FeatureMapFields.X_RN.value, FeatureMapFields.X_CIGAR.value, "rq", "tm"],
    "boolean_fields": [
        FeatureMapFields.IS_FORWARD.value,
        FeatureMapFields.IS_DUPLICATE.value,
    ],
    "fields_to_write_once": [
        FeatureMapFields.READ_COUNT.value,
        FeatureMapFields.FILTERED_COUNT.value,
        FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value,
        FeatureMapFields.HMER_CONTEXT_REF.value,
        FeatureMapFields.HMER_CONTEXT_ALT.value,
        FeatureMapFields.PREV_1.value,
        FeatureMapFields.PREV_2.value,
        FeatureMapFields.PREV_3.value,
        FeatureMapFields.NEXT_1.value,
        FeatureMapFields.NEXT_2.value,
        FeatureMapFields.NEXT_3.value,
    ],
    "boolean_fields_to_write_once": [
        FeatureMapFields.IS_CYCLE_SKIP.value,
    ],
}


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"


def write_a_pileup_record(
    record_dict: dict,
    rec_id: tuple,
    out_fh: pysam.VariantFile,
    header: pysam.VariantHeader,
    min_qual: int,
    qual_agg_func: callable = np.max,
    sample_name: str = "SAMPLE",
):
    """
    Write a pileup record to a vcf file

    Inputs:
        record_dict (dict): A dictionary containing the record information
        rec_id (str): The record id
        out_fh (pysam.VariantFile): A pysam file handle of the output vcf file
        header (pysam.VariantHeader): A pysam VarianyHeader pbject of the output vcf file
        min_qual (int): The minimum quality threshold
        qual_agg_func (function): The function to aggregate the quality scores (default: np.max)
        sample_name (str): The name of the sample (default: "SAMPLE")
    Output:
        rec (pysam.VariantRecord): A pysam VariantRecord object

    """
    format_fields = [GT, AD, DP, VAF]
    rec = header.new_record()
    rec.chrom = rec_id[0]
    rec.pos = rec_id[1]
    rec.ref = rec_id[2]
    rec.alts = rec_id[3]
    rec.id = "."
    rec.qual = qual_agg_func(record_dict[FeatureMapFields.X_QUAL.value])
    # INFO fields to aggregate
    # exceptions: X_QUAL
    rec.info[FeatureMapFields.X_QUAL.value] = list(record_dict[FeatureMapFields.X_QUAL.value])
    for field in fields_to_collect["numeric_array_fields"]:
        if field in record_dict:
            rec.info[field] = list(record_dict[field])
    for field in fields_to_collect["string_list_fields"]:
        if field in record_dict:
            rec.info[field] = "|".join(record_dict[field])
    for field in fields_to_collect["boolean_fields"]:
        if field in record_dict:
            rec.info[field] = "|".join(["T" if f else "F" for f in record_dict[field]])
    for field in fields_to_collect["fields_to_write_once"]:
        if field in record_dict:
            if [f == record_dict[field][0] for f in record_dict[field]].count(False) == 0:
                rec.info[field] = record_dict[field][0]
    for field in fields_to_collect["boolean_fields_to_write_once"]:
        if field in record_dict:
            if [f == record_dict[field][0] for f in record_dict[field]].count(False) == 0:
                rec.info[field] = record_dict[field][0]

    # FORMAT fields to aggregate
    record_dict[DP] = rec.info[FeatureMapFields.FILTERED_COUNT.value]
    record_dict[VAF] = record_dict["rec_counter"] / record_dict[DP]
    # record_dict[GT] = (0, 1)
    record_dict[GT] = (".", ".")
    record_dict[AD] = [
        record_dict[DP] - record_dict["rec_counter"],
        record_dict["rec_counter"],
    ]
    for format_field in format_fields:
        rec.samples[sample_name][format_field] = record_dict[format_field]
    # FILTER field
    if rec.qual < min_qual:
        curr_filter = FeatureMapFilters.LOW_QUAL.value
    elif rec.qual >= min_qual and record_dict["rec_counter"] == 1:
        curr_filter = FeatureMapFilters.SINGLE_READ.value
    else:
        curr_filter = FeatureMapFilters.PASS.value
    rec.filter.add(curr_filter)
    # write to file
    out_fh.write(rec)
    return rec


def pileup_featuremap(
    featuremap: str, output_vcf: str, genomic_interval: str = None, min_qual: int = 0, sample_name: str = "SAMPLE"
):
    """
    Pileup featuremap vcf to a regular vcf, save the aggregated quality and read count

    Inputs:
        featuremap (str): The input featuremap vcf file
        output_vcf (str): The output pileup vcf file
        genomic_interval (str): The genomic interval to pileup, format: chr:start-end
        min_qual (int): The minimum quality threshold
        sample_name (str): The name of the sample (default: "SAMPLE")
    Output:
        output_vcf (str): The output vcf file
    """
    cons_dict = defaultdict(dict)
    # process genomic interval input
    if genomic_interval is None:
        chrom = None
        start = None
        end = None
    else:
        assert re.match(r"^\w+:\d+-\d+$", genomic_interval), "Input genomic_interval format should be 'chrom:start-end'"
        genomic_interval_list = genomic_interval.split(":")
        chrom = genomic_interval_list[0]
        start = int(genomic_interval_list[1].split("-")[0])
        end = int(genomic_interval_list[1].split("-")[1])

    # generate a new header
    orig_header = pysam.VariantFile(featuremap).header
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.filters.add(
        "SingleRead", None, None, "Aggregated quality above threshold, Only a single read agrees with alternative"
    )
    header.filters.add("LowQual", None, None, "Aggregated quality below threshold")
    for field in fields_to_collect["numeric_array_fields"]:
        assert field in orig_header.info, f"Field {field} not found in the input featuremap vcf"
        header.info.add(field, ".", orig_header.info[field].type, orig_header.info[field].description)
    for field in fields_to_collect["string_list_fields"]:
        assert field in orig_header.info, f"Field {field} not found in the input featuremap vcf"
        header.info.add(field, "1", orig_header.info[field].type, orig_header.info[field].description)
    for field in fields_to_collect["boolean_fields"]:
        assert field in orig_header.info, f"Field {field} not found in the input featuremap vcf"
        header.info.add(field, "1", "String", orig_header.info[field].description)
    for field in fields_to_collect["fields_to_write_once"]:
        assert field in orig_header.info, f"Field {field} not found in the input featuremap vcf"
        header.info.add(field, "1", orig_header.info[field].type, orig_header.info[field].description)
    for field in fields_to_collect["boolean_fields_to_write_once"]:
        assert field in orig_header.info, f"Field {field} not found in the input featuremap vcf"
        header.info.add(field, "0", "Flag", orig_header.info[field].description)
    # exceptions: X_QUAL
    header.info.add(FeatureMapFields.X_QUAL.value, ".", "Float", "Quality of reads containing this location")
    header.formats.add(GT, 1, "String", "Genotype")
    header.formats.add(DP, 1, "Integer", "Read depth")
    header.formats.add(AD, "R", "Integer", "Allelic depths")
    header.formats.add(VAF, 1, "Float", "Allele frequency")
    # copy contigs from original header
    for contig in orig_header.contigs:
        header.contigs.add(contig)
    header.add_sample(sample_name)

    # open an output file
    out_fh = pysam.VariantFile(output_vcf, "w", header=header)

    cons_dict = defaultdict(dict)
    with pysam.VariantFile(featuremap) as f:
        prev_key = tuple()
        for rec in f.fetch(chrom, start, end):
            rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
            if "rec_counter" not in cons_dict[rec_id]:
                if len(cons_dict.keys()) > 1:
                    # write to file
                    write_a_pileup_record(cons_dict[prev_key], prev_key, out_fh, header, min_qual)
                    cons_dict.pop(prev_key)
                # initialize rec_counter
                cons_dict[rec_id]["rec_counter"] = 0
                # exceptions: X_QUAL
                cons_dict[rec_id][FeatureMapFields.X_QUAL.value] = np.array([])
                for field in fields_to_collect["numeric_array_fields"]:
                    cons_dict[rec_id][field] = np.array([])
                for field in fields_to_collect["string_list_fields"]:
                    cons_dict[rec_id][field] = []
                for field in fields_to_collect["boolean_fields"]:
                    cons_dict[rec_id][field] = []
                for field in fields_to_collect["fields_to_write_once"]:
                    cons_dict[rec_id][field] = []
                for field in fields_to_collect["boolean_fields_to_write_once"]:
                    cons_dict[rec_id][field] = []

            # update the record
            cons_dict[rec_id]["rec_counter"] += 1
            # exceptions: X_QUAL
            cons_dict[rec_id][FeatureMapFields.X_QUAL.value] = np.append(
                cons_dict[rec_id][FeatureMapFields.X_QUAL.value], rec.qual
            )
            for field in fields_to_collect["numeric_array_fields"]:
                cons_dict[rec_id][field] = np.append(cons_dict[rec_id][field], rec.info[field])
            for field in fields_to_collect["string_list_fields"]:
                cons_dict[rec_id][field] += [rec.info.get(field, ".")]
            for field in fields_to_collect["boolean_fields"]:
                cons_dict[rec_id][field] += [rec.info.get(field, False)]
            for field in fields_to_collect["fields_to_write_once"]:
                cons_dict[rec_id][field] += [rec.info[field]]
            for field in fields_to_collect["boolean_fields_to_write_once"]:
                cons_dict[rec_id][field] += [rec.info.get(field, False)]
            prev_key = rec_id

        # write last record
        write_a_pileup_record(cons_dict[prev_key], prev_key, out_fh, header, min_qual)

    out_fh.close()
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf
