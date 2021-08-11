import sys
import pandas as pd
from os.path import dirname, basename, join as pjoin
import pyfaidx
import logging


logger = logging.getLogger("mrd_utils")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


path = dirname(dirname(dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)


def create_control_signature(
    signature_bed_file, reference_fasta, signature_bed_file_control_output=None
):
    """
    Creates a control signature that matches each SNP in the input signature bed file with an adjacent position with
    the same trinucleotide motif, maintaining the same ref and alt composition. Non-SNP entries are ignored.

    Parameters
    ----------
    signature_bed_file:
        Input bed file
    reference_fasta:
        Reference fasta file
        an index (.fai) file is expected to be in the same path
    signature_bed_file_control_output:
        Output path, if None (default) the input file name is used with a ".control.bed" suffix

    Returns
    -------

    """
    ref = pyfaidx.Fasta(reference_fasta)

    if signature_bed_file_control_output is None:
        signature_bed_file_control_output = f"{signature_bed_file}.control.bed".replace(
            "bed.control.bed", ".control.bed"
        )

    df_signature = pd.read_csv(
        signature_bed_file,
        sep="\t",
        header=None,
        usecols=range(11),
        names=[
            "chrom",
            "chromStart",
            "chromEnd",
            "id",
            "quality",
            "ref",
            "alt",
            "filter",
            "info",
            "format",
            "sample",
        ],
    )
    out_rows = list()
    for _, row in df_signature.iterrows():
        if (row["chromEnd"] - row["chromStart"] != 1) or len(
            row["ref"]
        ) != 1:  # not a snp
            continue
        chrom = row["chrom"]
        motif = ref[chrom][row["chromStart"] - 1 : row["chromStart"] + 2].seq
        assert row["ref"] == motif[1], f"Inconsistency in reference found!\n{row}"

        row_out = row.copy()
        pos = row["chromStart"] + 5  # starting position
        delta = 1
        while True:
            new_motif = ref[chrom][pos - 1 : pos + 2].seq
            if (
                len(new_motif) < 3 or new_motif == "NNN"
            ):  # reached the end of the chromosome or a reference gap
                if (
                    delta == -1
                ):  # if we already looked in the other direction, stop and give up this entry (exceedingly rare)
                    break
                # start looking in the other direction
                delta = -1
                pos = row["chromStart"] - 5
                continue
            if motif == new_motif:  # motif matches
                row_out["chromStartOriginal"] = row_out["chromStart"]  # keep original
                row_out["chromStart"] = pos
                row_out["chromEnd"] = pos + 1
                out_rows.append(row_out)
                break
            pos += delta
    df_out = pd.concat(out_rows, axis=1).T
    df_out.to_csv(signature_bed_file_control_output, sep="\t", header=None, index=False)
