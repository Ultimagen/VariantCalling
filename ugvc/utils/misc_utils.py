from __future__ import annotations

import collections
import itertools
import pkgutil
from os.path import dirname
from os.path import join as pjoin

import numpy as np
import pysam
from matplotlib import pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 26
TITLE_SIZE = 36
FIGSIZE = (16, 8)
GRID = True


def runs_of_one(array, axis=None):
    """
    returns start and end (half open) of intervals of ones in a binary vector
    if axis=None - runs are identified according to the (only) non-singleton axis
    """
    array = array.astype(np.int8)
    if isinstance(array, np.ndarray):
        array = np.array(array)
    if not axis:
        shapes = [x for x in array.shape if x != 1]
        if len(shapes) != 1:
            raise RuntimeError("runs_of_one - too many non-singleton axes in array")
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


def searchsorted2d(ar_a: np.ndarray, ar_b: np.ndarray) -> np.ndarray:
    """
    Inserts ith element of b into sorted ith row of a

    Parameters
    ----------
    ar_a: np.ndarray
            rxc matrix, each rows is sorted
    ar_b: np.ndarray
            rx1 vector

    Returns
    -------
    np.ndarray
            rx1 vector of locations
    """
    dim1_a, dim2_a = ar_a.shape
    ar_b = ar_b.ravel()
    assert ar_b.shape[0] == ar_a.shape[0], "Number of values of array b equal number of rows of array a"
    max_num = np.maximum(ar_a.max() - ar_a.min(), ar_b.max() - ar_b.min()) + 1
    r_seq = max_num * np.arange(ar_a.shape[0])
    indices = np.searchsorted(((ar_a.T + r_seq).T).ravel(), ar_b + r_seq)
    return indices - dim2_a * np.arange(dim1_a)


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
        with open(output_file, "w", encoding="ascii") as outfile:
            lengths = infile.header.lengths
            contigs = infile.header.references
            for contig, length in zip(contigs, lengths):
                outfile.write(f"{contig}\t{length}\n")


def max_merits(specificity, recall):
    """Finds ROC envelope from multiple sets of specificity and recall"""
    N = specificity.shape[0]
    ind_max = np.ones(N, np.bool)
    for j in range(N):
        for i in range(N):
            if (specificity[i] > specificity[j]) & (recall[i] > recall[j]):
                ind_max[j] = False
                continue
    ind = np.where(ind_max)[0]
    ind_sort_recall = np.argsort(recall[ind])
    return ind[ind_sort_recall]


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
    return interval[0] <= pos < interval[1]


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
    func: collections.abc.Callable,
    *args,
    exception_type: Exception = Exception,
    handle: collections.abc.Callable = lambda e: e,
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
    except exception_type as e:  # pylint: disable=broad-except
        return handle(e)


def set_pyplot_defaults(
    title_size=TITLE_SIZE,
    small_size=SMALL_SIZE,
    medium_size=MEDIUM_SIZE,
    bigger_size=BIGGER_SIZE,
    grid=GRID,
    figsize=FIGSIZE,
):
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=title_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=bigger_size)  # fontsize of the x and y labels
    plt.rc("axes", grid=grid)  # is grid on
    plt.rc("xtick", labelsize=medium_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=medium_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=medium_size)  # legend fontsize
    plt.rc("figure", titlesize=title_size)  # fontsize of the figure title
    plt.rc("figure", figsize=figsize)  # size of the figure


def idx_last_nz(inp: np.ndarray | list) -> np.ndarray:
    """Index of the closest previous nonzero element for each element in the array.
    If the array starts with 0 - the index is -1

    Parameters
    ----------
    inp : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
    """
    if not isinstance(inp, np.ndarray):
        inp = np.array(inp)
    nzs = np.concatenate(([-1], np.nonzero(inp)[0]))
    nzcounts = np.cumsum(inp > 0)
    return nzs[nzcounts]


def idx_next_nz(inp: np.ndarray | list) -> np.ndarray:
    """Index of the closest next nonzero element for each element in the array.
    If the array starts with 0 - the index is len(input)

    Parameters
    ----------
    inp : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
    """
    result = idx_last_nz(inp[::-1])
    result = len(inp) - result - 1
    return result[::-1]
