import itertools
import pkgutil
from os.path import dirname
from os.path import join as pjoin
from typing import Callable

import numpy as np
import pandas as pd
import pysam


def runs_of_one(array, axis=None):
    """
    returns start and end (half open) of intervals of ones in a binary vector
    if axis=None - runs are identified according to the (only) non-singleton axis
    """
    array = array.astype(np.int8)
    if isinstance(array, np.ndarray):
        array = np.array(array)
    if not axis:
        sh = [x for x in array.shape if x != 1]
        if len(sh) != 1:
            raise RuntimeError("runs_of_one - too many non-singleton axes in array")
        else:
            array = np.squeeze(array).reshape(1, -1)
            axis = 1
    if axis != 1:
        array.reshape(array.shape[::-1])
    runs_of_ones = []
    for i in range(array.shape[0]):
        one_line = array[i, :]

        diffs = np.diff(one_line)

        starts = np.nonzero(diffs == 1)[0] + 1
        if one_line[0] == 1:
            starts = np.concatenate(([0], starts))
        ends = np.nonzero(diffs == -1)[0] + 1
        if one_line[-1] == 1:
            ends = np.concatenate((ends, [len(one_line)]))

        runs_of_ones.append(zip(starts, ends))

    return runs_of_ones


def searchsorted2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Inserts ith element of b into sorted ith row of a

    Parameters
    ----------
    a: np.ndarray
            rxc matrix, each rows is sorted
    b: np.ndarray
            rx1 vector

    Returns
    -------
    np.ndarray
            rx1 vector of locations
    """
    m, n = a.shape
    b = b.ravel()
    assert b.shape[0] == a.shape[0], "Number of values of b equal number of rows of a"
    max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * np.arange(a.shape[0])
    p = np.searchsorted(((a.T + r).T).ravel(), b + r)
    return p - n * np.arange(m)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def shiftarray(arr: np.ndarray, num: int, fill_value: np.float = np.nan) -> np.ndarray:
    """Shifts array by num to the right

    Parameters
    ----------
    arr: np.ndarray
        Array to be shifted
    num: int
        Shift size (negative - left shift)
    fill_value: np.float
        Fill value
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def contig_lens_from_bam_header(bam_file: str, output_file: str):
    """Creates a "sizes" file from contig lengths in bam header.
    Sizes file is per the UCSC spec: contig <tab> length

    Parameters
    ----------
    bam_file: str
        Bam file
    output_file: str
        Output file

    Returns
    -------
    None, writes output_file
    """

    with pysam.AlignmentFile(bam_file) as infile:
        with open(output_file, "w") as outfile:
            lengths = infile.header.lengths
            contigs = infile.header.references
            for c, l in zip(contigs, lengths):
                outfile.write(f"{c}\t{l}\n")


def max_merits(specificity, recall):
    """Finds ROC envelope from multiple sets of specificity and recall
    """
    N = specificity.shape[0]
    ind_max = np.ones(N, np.bool)
    for j in range(N):
        for i in range(N):
            if (specificity[i] > specificity[j]) & (recall[i] > recall[j]):
                ind_max[j] = False
                continue
    ind = np.where(ind_max)[0]
    a = np.argsort(recall[ind])
    return ind[a]


def isin(pos: int, interval: tuple) -> bool:
    """Is position inside the [interval)

    Parameters
    ----------
    pos: int
        Position
    interval: tuple
        [start,end)

    Returns
    -------
    bool
    """
    return pos >= interval[0] and pos < interval[1]


def find_scripts_path() -> str:
    """Locates the absolute path of the scripts installation

    Parameters
    ----------
    None

    Returns
    -------
    str
        The path
    """
    package = pkgutil.get_loader("ugvc")
    return pjoin(dirname(package.get_filename()), "bash")


def catch(
    func: Callable,
    *args,
    exception_type: Exception = Exception,
    handle: Callable = lambda e: e,
    **kwargs,
):
    """From https://stackoverflow.com/a/8915613 general wrapper that
    catches exception and returns value. Useful for handling error in list comprehension

    Parameters
    ----------
    func: Callable
        Function being wrapped
    *args:
        func non-named arguments
    exception_type:
        Type of exception to catch
    handle:
        How to handle the exception (what value to return)
    **kwargs:
        func named arguments
    """
    try:
        return func(*args, **kwargs)
    except exception_type as e:
        return handle(e)