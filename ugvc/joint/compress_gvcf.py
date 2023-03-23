import pysam
import argparse
from typing import List
import sys

def get_sample_value(record, key, sample_name):
    return record.samples[sample_name].values()[record.format.keys().index(key)] if key in record.samples[
        sample_name].keys() else None


def get_compressed_pl_into_3_values(record, sample_name):
    '''
    This function takes out PL values out of record, sample_name
    PL is tuple which is size is decided by the number of alts.
    This function makes it a 3 tuple for the merged record in the following manner:
    The input PL is of the form of
    (0,0),(0,1)(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)..
    so this function makes a 3 tuple by:
    ((0,0), min((0,1),(0,2),(0,3)..),min(all the others which are not 0/*)
    Parameters
    ----------
    record
    sample_name

    Returns
    -------

    '''
    compressed_pl = []
    pl = get_sample_value(record, 'PL', sample_name)
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



def get_compressed_pl_into_3_values(pl):
    '''
    PL is tuple which its size is decided by the number of alts.
    This function makes it a 3 tuple for the merged record in the following manner:
    The input PL is of the form of
    (0,0),(0,1)(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)..
    so this function makes a 3 tuple by:
    ((0,0), min((0,1),(0,2),(0,3)..),min(all the others which are not 0/*)
    Parameters
    ----------
    pl

    Returns
    -------
    3-values tuple
    '''
    if len(pl) == 3:
        return pl
    else:
        compressed_pl = []
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
    ap_var = argparse.ArgumentParser(prog="compress_gvcf.py", description="Compress GVCF file by merging similar rows")
    ap_var.add_argument("--input_path", help="Input gvcf file path", required=True, type=str)
    ap_var.add_argument("--output_path", help="Output gvcf file path", required=True, type=str)
    return ap_var


def compressGVCF(input_path, output_path):
    """Makes the g.vcf file smaller by merging similar sequential records
    The mergining is doe in the following way:
    We merge sequential records as long as there GQ value of all the records in this merge is not different
    from each other by more than 10
    In addition, we want to keep RefCall records with GQ<22 and not merge them.
    When merging we create a new record with the following parameters:
    - chrom, pos, ref: are of the first record in this group of merged records
    - id: '.'
    - alts: ["<*>"]
    - qual: 0
    - GT: (0, 0)
    - GQ: is the avg GQ of all the merged records
    - MIN_DP: is the minimum MIN_DP of all the merged records
    - END: is the end of the last record in the merged records
    - PL: it is always of size 3, and each value is the minimum value in this index among all the records,
        see get_compressed_pl_into_3_values for more details

    Parameters
    ----------
    input_path : str
        input gvcf
    output_path : str
        compressed gvcf

    Returns
    -------
    tuple(int,int)
        Number of records in input g.vcf, number of records in compressed g.vcf

    """

    gvcf_in = pysam.VariantFile(input_path, "r")
    gvcf_out = pysam.VariantFile(output_path, "w", header=gvcf_in.header)
    fetch_gvcf = gvcf_in.fetch()
    sample_name = list(gvcf_in.header.samples)[0]
    merge_size = 0
    prev_record = None
    first_record = None
    count_in = 0
    count_out = 0
    for index, record in enumerate(fetch_gvcf):
        count_in += 1
        if prev_record is None:
            # first record
            prev_record = record
            first_record = record
            min_gq = get_sample_value(record, 'GQ', sample_name)
            max_gq = min_gq
            gq_sum = min_gq
            min_dp = get_sample_value(record, 'MIN_DP', sample_name)
            merge_size += 1
            pl = get_compressed_pl_into_3_values(get_sample_value(record, 'PL', sample_name))
            continue

        gq_value = get_sample_value(record, 'GQ', sample_name)
        is_refcall = ('RefCall' in record.filter.keys())
        prev_gq_value = get_sample_value(prev_record, 'GQ', sample_name)
        is_prev_refcall = ('RefCall' in prev_record.filter.keys())

        if (is_refcall and (gq_value <= 22)) or (record.chrom != prev_record.chrom) or (gq_value - min_gq >= 10) or (
                max_gq - gq_value >= 10) or (is_prev_refcall and (prev_gq_value <= 22)):
            # should write prev_record
            # if we don't merge - just add the record to the gvcf
            if merge_size == 1:
                gvcf_out.write(prev_record)
                count_out += 1
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
                count_out += 1
            # get ready for the next iteration
            prev_record = record
            first_record = record
            merge_size = 1
            min_gq = get_sample_value(record, 'GQ', sample_name)
            max_gq = min_gq
            gq_sum = min_gq
            min_dp = get_sample_value(record, 'MIN_DP', sample_name)
            pl = get_compressed_pl_into_3_values(get_sample_value(record, 'PL', sample_name))
        else:
            # just continue with the same set of lines
            prev_record = record
            merge_size += 1
            min_gq = min(min_gq, gq_value)
            max_gq = max(max_gq, gq_value)
            cur_min_dp = get_sample_value(record, 'MIN_DP', sample_name)
            if cur_min_dp is not None:
                if min_dp is not None:
                    min_dp = min(min_dp, cur_min_dp)
                else:
                    min_dp = cur_min_dp
            cur_pl = get_compressed_pl_into_3_values(get_sample_value(record, 'PL', sample_name))
            pl = (min(cur_pl[0], pl[0]), min(cur_pl[1], pl[1]), min(cur_pl[2], pl[2]))
            gq_sum += gq_value

    gvcf_in.close()
    gvcf_out.close()
    pysam.tabix_index(output_path, preset='vcf')

    return count_in, count_out


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    result = compressGVCF(args.input_path, args.output_path)
    sys.stderr.write(f"Compressed {result[0]} into {result[1]} records")