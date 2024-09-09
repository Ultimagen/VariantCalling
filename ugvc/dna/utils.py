from __future__ import annotations

import re
from os.path import isfile

import numpy as np
import pyfaidx

# from typing import Union


def revcomp(seq: str | list | np.ndarray) -> str | list | np.ndarray:
    """Reverse complements DNA given as string

    Parameters
    ----------
    :param: seq Union[str,list,np.ndarray]
        DNA string
    :raises ValueError: is seq is not of the right type

    :return: str | list | np.ndarray


    """
    complement = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
    }
    if isinstance(seq, str):
        reverse_complement = "".join(complement.get(base, base) for base in reversed(seq))
    elif isinstance(seq, list):
        reverse_complement = [complement.get(base, base) for base in reversed(seq)]
    elif isinstance(seq, np.ndarray):
        reverse_complement = np.array([complement.get(base, base) for base in reversed(seq)])
    else:
        raise ValueError(f"Got unexpected variable {seq} of type {type(seq)}, expected str, list or numpy array")

    return reverse_complement


def hmer_length(seq: pyfaidx.Sequence, start_point: int) -> int:
    """Return length of hmer starting at point start_point

    Parameters
    ----------
    seq: pyfaidx.Sequence
        Sequence
    start_point: int
        Starting point

    Returns
    -------
    int
        Length of hmer (at least 1)
    """

    idx = start_point
    while seq[idx].seq.upper() == seq[start_point].seq.upper():
        idx += 1
    return idx - start_point


def get_chr_sizes(sizes_file: str) -> dict:
    """Returns dictionary from chromosome name to size

    Parameters
    ----------
    sizes_file: str
        .sizes file (use e.g.  cut -f1,2 Homo_sapiens_assembly19.fasta.fai > Homo_sapiens_
        assembly19.fasta.sizes to generate), .fai file or .dict file.
        Any file which doesnot end with fai or dict will be considered .sizes

    Returns
    -------
    dict:
        Dictionary from name to size
    """

    assert isfile(sizes_file), f"Input_file {sizes_file} not found"
    if sizes_file.endswith("dict"):
        chrom_sizes = {}
        with open(sizes_file, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("@SQ"):
                    fields = line[3:].strip().split("\t")
                    row = {}  # Dict of all fields in the row
                    for field in fields:
                        key, value = field.split(":", 1)
                        row[key] = value
                    chrom_sizes[row["SN"]] = int(row["LN"])
        return chrom_sizes
    with open(sizes_file, encoding="ascii") as sizes:
        tmp = [x.strip().split()[:2] for x in sizes]
    return dict((x[0], int(x[1])) for x in tmp)


def get_max_softclip_len(cigar):
    group = re.match("(?P<start>[0-9]+S)?[0-9]+[0-9MID]+[MID](?P<end>[0-9]+S)?", cigar).groups()
    start = int(group[0][:-1]) if group[0] else 0
    end = int(group[1][:-1]) if group[1] else 0
    return max(start, end)
