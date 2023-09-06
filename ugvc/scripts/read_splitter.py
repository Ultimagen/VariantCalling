#!/env/python
# -*- coding: utf-8 -*-
"""
This script splits single end reads from FASTQ into two files of paired end reads
"""

import sys

import tqdm
from Bio import SeqIO


def main(argc, argv):
    if argc != 3:
        print("Usage: python split_single_end_reads.py <input_file>.fq <output_prefix>", file=sys.stderr)
        return 1
    handle = SeqIO.parse(argv[1], "fastq")
    with open(argv[2] + "_1.fq", "w", encoding="utf8") as out1:
        with open(argv[2] + "_2.fq", "w", encoding="utf8") as out2:
            for record in tqdm.tqdm(handle):
                splitlen = len(record.seq) // 2
                qual = record.letter_annotations["phred_quality"]

                record1 = SeqIO.SeqRecord(record.seq[:splitlen], record.id + "/1")
                record1.letter_annotations["phred_quality"] = qual[:splitlen]

                record1.description = ""
                out1.write(record1.format("fastq"))
                record2 = SeqIO.SeqRecord(record.seq[splitlen:].reverse_complement(), record.id + "/2")
                record2.letter_annotations["phred_quality"] = qual[splitlen:][::-1]

                record2.description = ""
                out2.write(record2.format("fastq"))
    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
