import math

import BeadsData
import numpy as np
import pandas as pd

from ugvc.dna.format import DEFAULT_FLOW_ORDER

from . import utils

_workdir = "/home/ilya/proj/VariantCalling/work/190605/"

error_model = pd.read_excel(
    "/home/ilya/proj/VariantCalling/work/190623/error_model.xls", index_col=0
)
error_model[np.isnan(error_model)] = 0.15


def getQualityStringMer(hmer: int, base: chr) -> str:
    """Returns quality **string** for hmer estimated from error model.
    Specifically for hmer k will return IIIII.II (k-1 times)? where ? is a
    FASTQ formatted quality based on `quality_scores.error_model`

    Parameters
    ----------
    hmer: int
        Count
    base: chr

    Returns
    -------
    str: A Phred-based FASTQ formatted quality string
    """

    if hmer == 0:
        return ""
    error_prob = error_model.loc[min(hmer, 8), base]
    phred = int(-10 * math.log10(error_prob))
    phred_symbol = chr(33 + phred)
    other = chr(33 + 30)
    return other * (hmer - 1) + phred_symbol


def getQualityVals(hmer: int, base: chr) -> list:
    """Returns quality **list** for hmer estimated from error model.
    Specifically for hmer k will return [30,30,...] (k-1 times)[x] where x is a
    quality in Phred based on `quality_scores.error_model`

    Parameters
    ----------
    hmer: int
        Count
    base: chr

    Returns
    -------
    list: A list of qualities

    See also
    --------
    `getQualityStringMer`
    """

    error_prob = error_model.loc[hmer, base]
    phred = int(-10 * math.log10(error_prob))
    other = 30
    return [other] * (hmer - 1) + [phred]


def getQualityStringRead(read: str) -> str:
    """Returns FASTQ formatted quality string for read. Qualities are
    generated hmer by hmer from the error model

    Parameters
    ----------
    read: str
        Read sequence

    Returns
    -------
    str
        Quality string

    See also
    --------
    `getQualityStringMer`
    """

    key = BeadsData.BeadsData.generateKeyFromSequence(read, truncate=None)
    base_order = (DEFAULT_FLOW_ORDER * len(key))[: len(key)]
    qualities = "".join(
        [getQualityStringMer(key[i], base_order[i]) for i in range(len(key))]
    )
    return qualities


def getQualityVector(read: str) -> list:
    """Returns Phread list of qualities for the read. Qualities are
    generated hmer by hmer from the error model

    Parameters
    ----------
    read: str
        Read sequence

    Returns
    -------
    list
        Qualities list

    See also
    --------
    `getQualityStringMer`
    """

    key = BeadsData.BeadsData.generateKeyFromSequence(read)
    base_order = (DEFAULT_FLOW_ORDER * len(key))[: len(key)]
    qualities = sum(
        [getQualityVals(key[i], base_order[i]) for i in range(len(key))], []
    )
    return qualities


def addQualitiesFasta(input_file: str, output_file: str) -> None:
    """Converts FASTA to FASTQ with qualities generated from the error model

    Parameters
    ----------
    input_file: str
        FASTA input file name
    output_file: str
        FASTQ output file name

    Returns
    -------
    None, writes into `output_file`

    See also
    --------
    `getQualityStringRead`
    """
    with open(input_file) as inp:
        with open(output_file, mode="w") as outp:
            for (rn, rs) in utils.grouper(map(lambda x: x.strip(), inp), 2):
                qual = getQualityStringRead(rs)
                assert len(qual) == len(rs), "Quality shorter!"
                outp.write("@{}\n".format(rn[1:]))
                outp.write("{}\n".format(rs))
                outp.write("+\n")
                outp.write(qual + "\n")
