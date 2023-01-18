# pylint: disable=duplicate-code
from __future__ import annotations

import numpy as np

def safe_divide(numerator: float, denominator: float, return_if_denominator_is_0: int = 0):
    """

    Parameters
    ----------
    numerator : float
        numerator
    denominator : float
        denominator
    return_if_denominator_is_0 : boolean
        return value whenever denominator is 0 (undivisable)

    Returns
    -------
    if denominator != 0 -> numerator/denominator
    else -> return_if_denominator_is_0
    """
    if denominator == 0:
        return return_if_denominator_is_0

    return numerator / denominator


def phred(p: tuple[int] | np.ndarray) -> np.ndarray:
    """
    Transform probablitied to Phred quality scores
    See https://en.wikipedia.org/wiki/Phred_quality_score
    """
    q = -10 * np.log10(np.array(p, dtype=np.float))
    return q


def unphred(q: tuple[int] | np.ndarray) -> np.ndarray:
    """
    Transform Phred quality scores to probablitied
    See https://en.wikipedia.org/wiki/Phred_quality_score
    """
    p = np.power(10, -np.array(q, dtype=np.float) / 10)
    return p
