#!/bin/env python
import argparse

import numpy as np
import pyfaidx
import tqdm.auto as tqdm
from numpy import random
from ugbio_core.bed_writer import parse_intervals_file
from ugbio_core.flow_format import flow_based_read as fbr

from ugvc.vcfbed import interval_file


# this code goes over reference genome and collects the table of homopolymers
# it is used to generate the table of homopolymers for the reference genome
def run():
    parser = argparse.ArgumentParser(description="Collect homopolymer locations")
    parser.add_argument("--reference", help="Reference genome", required=True, type=str)
    parser.add_argument(
        "--collection_regions",
        help="bed or interval_list file with regions to collect the info from",
        required=True,
        type=str,
    )
    parser.add_argument("--output", help="Homopolymer table", type=str, required=True)
    parser.add_argument("--max_hpol_length", default=20, help="Maximum homopolymer length to consider", type=int)
    parser.add_argument(
        "--max_number_to_collect",
        default=100000,
        help="Maximum number of homopolymer type (nucleotide, length) to collect",
        type=int,
    )
    args = parser.parse_args()

    sampling_fractions = plan_sampling(args.collection_regions)

    hpol_table = collect_homopolymers(
        args.reference, args.collection_regions, args.max_hpol_length, args.max_number_to_collect, sampling_fractions
    )
    write_hpol_table(hpol_table, args.output)


def plan_sampling(collection_regions: str) -> list[float]:
    """For every interval in the collection region, calculate what is its length as a
    fraction of the total length of the collection region. This fraction will be used to sample

    Parameters
    ----------
    collection_regions : str
        Path to the collection regions file (bed or interval_list)

    Returns
    -------
    list
        List of fractions of the length the same as the length of the interval list
    """
    ilst = interval_file.IntervalFile(interval=collection_regions)
    bed_ilst = ilst.as_bed_file()
    df = parse_intervals_file(bed_ilst, sort=False)
    total_length = (df["end"] - df["start"]).sum()
    df["fraction"] = (df["end"] - df["start"]) / total_length
    return df["fraction"].tolist()


def collect_homopolymers(
    reference: str,
    collection_regions: str,
    max_hpol_length: int,
    max_number_to_collect: int,
    sampling_fractions: list[float],
) -> list[tuple]:
    """Collect homopolymers from the reference genome

    Parameters
    ----------
    reference : str
        Path to the reference genome
    collection_regions: str
        Bed or interval_list file
    max_hpol_length : int
        Maximum homopolymer length to consider
    max_number_to_collect : int
        Maximum number of homopolymer type (nucleotide, length) to collect
    sampling_fractions : list[float]
        Fractions of the length of the collection region

    Returns
    -------
    list
        List of tuples (chromosome, position, length, nucleotide)
    """
    hpol_list = []
    ref_fasta = pyfaidx.Fasta(reference)
    ilst = interval_file.IntervalFile(interval=collection_regions)
    df = parse_intervals_file(ilst.as_bed_file(), sort=False)
    for i, ix in tqdm.tqdm(enumerate(df.index)):
        seq = ref_fasta[df.loc[ix, "chromosome"]][df.loc[ix, "start"] : df.loc[ix, "end"]]
        try:
            key = fbr.generate_key_from_sequence(seq.seq, "TGCA", non_standard_as_a=True)
        except ValueError:
            print("Skipping due to non-standard nucleotides")
            continue
        k2base = np.cumsum(key)

        for h in range(1, max_hpol_length + 1):
            for j in range(len("TGCA")):
                mask = np.zeros(len(key), dtype=bool)
                mask[j :: len("TGCA")] = True
                select = (key == h) & (mask)
                locs = np.nonzero(select)[0]
                random.shuffle(locs)
                take = int(np.ceil(sampling_fractions[i] * max_number_to_collect))
                locs = locs[:take]
                bases = k2base[locs] + df.loc[ix, "start"]
                hpol_list.extend([(df.loc[ix, "chromosome"], b, h, "TGCA"[j]) for b in bases])
    hpol_list = sorted(hpol_list, key=lambda x: (x[0], x[1]))
    return hpol_list


def write_hpol_table(hpol_list: list[tuple], output: str):
    """Write homopolymer table to the output TSV file (output) in the format
    chromosome<tab>position<tab>length<tab>nucleotide

    Parameters
    ----------
    hpol_list : list[tuple]
        List of homopolymer locations found by collect_homopolymers
    output : str
        Path to the output file
    """
    with open(output, "w", encoding="utf-8") as fh:
        for locus in tqdm.tqdm(hpol_list):
            chromosome, position, length, nucleotide = locus
            fh.write(f"{chromosome}\t{position}\t{length}\t{nucleotide}\n")


if __name__ == "__main__":
    run()
