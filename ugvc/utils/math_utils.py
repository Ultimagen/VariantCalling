from __future__ import annotations


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
