import pysam
import argparse
from typing import Any, List, Optional, Tuple
import sys

def get_sample_value(record, key, sample_name):
    return record.samples[sample_name].values()[record.format.keys().index(key)] if key in record.samples[
        sample_name].keys() else None


def get_compressed_pl_into_3_values(record):
    compressed_pl = []
    pl = get_sample_value(record, 'PL')
    if len(pl) == 3:
        return pl
    else:
        n = 0
        sum_n = 0
        while sum_n < len(pl):
            cur_compressed_pl = pl[sum_n:sum_n + n + 1]
            n += 1
            sum_n += n
            for i in range(len(cur_compressed_pl)):
                if i + 1 >= len(compressed_pl) and i <= 1:
                    compressed_pl.append(cur_compressed_pl[i])
                else:
                    compressed_pl[min(i + 1, 2)] = min(compressed_pl[min(i + 1, 2)], cur_compressed_pl[i])
        return tuple(compressed_pl)

def get_parser() -> argparse.ArgumentParser:
    ap_var = argparse.ArgumentParser(prog="run_comparison_pipeline.py", description="Compare VCF to ground truth")
    ap_var.add_argument("--input_path", help="Input gvcf file path", required=True, type=str)
    ap_var.add_argument("--output_path", help="Output gvcf file path", required=True, type=str)
    return ap_var


# def run(argv: list[str]):
def run(argv: List[str]):
    parser = get_parser()
    args = parser.parse_args(argv[1:])
    gvcf_in = pysam.VariantFile(args.input_path, "r")
    gvcf_out = pysam.VariantFile(args.output_path, "w", header=gvcf_in.header)
    fetch_gvcf = gvcf_in.fetch()
    sample_name = list(gvcf_in.header.samples)[0]
    merge_size = 0
    prev_record = None
    first_record = None
    for index, record in enumerate(fetch_gvcf):
        if prev_record is None:
            # first record
            prev_record = record
            first_record = record
            min_gq = get_sample_value(record, 'GQ')
            max_gq = min_gq
            gq_sum = min_gq
            min_dp = get_sample_value(record, 'MIN_DP')
            merge_size += 1
            pl = get_compressed_pl_into_3_values(record)
            continue

        gq_value = get_sample_value(record, 'GQ')
        is_refcall = ('RefCall' in record.filter.keys())
        prev_gq_value = get_sample_value(prev_record, 'GQ')
        is_prev_refcall = ('RefCall' in prev_record.filter.keys())

        if (is_refcall and (gq_value <= 22)) or (record.chrom != prev_record.chrom) or (gq_value - min_gq >= 10) or (
                max_gq - gq_value >= 10) or (is_prev_refcall and (prev_gq_value <= 22)):
            # should write prev_record
            # if we don't merge - just add the record to the gvcf
            if merge_size == 1:
                gvcf_out.write(prev_record)
            else:
                # create merged record
                new_record = pysam.VariantHeader.new_record(gvcf_in.header)
                new_record.chrom = first_record.chrom
                new_record.pos = first_record.pos
                new_record.id = "."
                new_record.ref = first_record.ref
                new_record.alts = ["<*>"]
                new_record.qual = 0
                new_record.stop = prev_record.stop
                new_record.samples[sample_name]['GT'] = (0, 0)
                new_record.samples[sample_name]['GQ'] = int(gq_sum / merge_size)
                if min_dp is not None:
                    new_record.samples[sample_name]['MIN_DP'] = min_dp
                new_record.samples[sample_name]['PL'] = pl
                gvcf_out.write(new_record)
            # get ready for the next iteration
            prev_record = record
            first_record = record
            merge_size = 1
            min_gq = get_sample_value(record, 'GQ')
            max_gq = min_gq
            gq_sum = min_gq
            min_dp = get_sample_value(record, 'MIN_DP')
            pl = get_compressed_pl_into_3_values(record)
        else:
            # just continue with the same set of lines
            prev_record = record
            merge_size += 1
            min_gq = min(min_gq, gq_value)
            max_gq = max(max_gq, gq_value)
            cur_min_dp = get_sample_value(record, 'MIN_DP')
            if cur_min_dp is not None:
                if min_dp is not None:
                    min_dp = min(min_dp, cur_min_dp)
                else:
                    min_dp = cur_min_dp
            cur_pl = get_compressed_pl_into_3_values(record)
            pl = (min(cur_pl[0], pl[0]), min(cur_pl[1], pl[1]), min(cur_pl[2], pl[2]))
            gq_sum += gq_value

    gvcf_in.close()
    gvcf_out.close()


if __name__ == "__main__":
    run(sys.argv)