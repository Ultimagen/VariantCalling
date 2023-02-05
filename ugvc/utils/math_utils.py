# pylint: disable=duplicate-code
"""Summary
"""
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
    return_if_denominator_is_0 : int, optional
        return value whenever denominator is 0 (undivisable)

    Returns
    -------
    if denominator != 0 -> numerator/denominator
    else -> return_if_denominator_is_0
    """
    if denominator == 0:
        return return_if_denominator_is_0

    return numerator / denominator


def phred(p: list[float] | tuple[float] | np.ndarray) -> np.ndarray:
    """
    Transform probablitied to Phred quality scores
    See https://en.wikipedia.org/wiki/Phred_quality_score

    Parameters
    ----------
    p : Union[list[float], tuple[float], np.ndarray]
        List of float probability values

    Returns
    -------
    np.ndarray
        List of float quality values
    """
    q = -10 * np.log10(np.array(p, dtype=np.float))
    return q


def phred_str(p: list[float] | tuple[float] | np.ndarray) -> str:
    """Convert list of error probabilities to phred-encoded string

    Parameters
    ----------
    p : Union[list[float], tuple[float], np.ndarray]
        List of float probability values

    Returns
    -------
    str
        Basequality string
    """
    q = phred(p)
    return "".join(chr(int(x) + 33) for x in q)


def unphred(q: list[int | float] | tuple[int | float] | np.ndarray) -> np.ndarray:
    """Transform Phred quality scores to probablities
    See https://en.wikipedia.org/wiki/Phred_quality_score

    Parameters
    ----------
    q : Union[list, tuple, np.ndarray]
        List of integer or float phred qualities

    Returns
    -------
    np.ndarray
        List of error probabilities
    """
    p = np.power(10, -np.array(q, dtype=np.float) / 10)
    return p


def unphred_str(strq: str) -> np.ndarray:
    """Converts string of qualities to array of error probabilities

    Parameters
    ----------
    strq : str
        BQ-like string

    Returns
    -------
    np.ndarray
        Array of error probabilities
    """
    q = [ord(x) - 33 for x in strq]
    return unphred(q)
