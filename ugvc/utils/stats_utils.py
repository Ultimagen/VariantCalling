from typing import List, Tuple

import numpy as np
from scipy.stats import multinomial

from ugvc.utils.math_utils import safe_divide


# Goodness of fit functions

def scale_contingency_table(table: List[int], n: int) -> List[int]:
    """
    Parameters
    ----------
    table: List[int]
        contingency table represented as list of counts
    n: int
        scale-factor
    Returns
    -------
    A contingency table where the sum of categories is approximately n
    """
    sum_table = sum(table)
    if sum_table > 0:
        scaled_table = np.array(table) * (n / sum_table)
        return list(np.round(scaled_table).astype(int))
    else:
        return table


def get_corrected_multinomial_frequencies(counts: List[int]) -> np.array:
    """

    Parameters
    ----------
    counts - List[int]
        contingency table represented as list of ints

    Returns
    -------
    Frequency of each category after add-one correction
    """
    corrected_counts = np.array(counts) + 1
    return corrected_counts / np.sum(corrected_counts)


def multinomial_likelihood(actual: List[int], expected: List[int]) -> float:
    """
    Calculate the likelihood of observing actual, under the multinomial distribution described by expected

    Parameters
    ----------
    actual : List[int]
        a list of counts (integer observations)
    expected : List[int]
        a list counts from which to fit a multinomial distribution to
    Returns
    -------
    Likelihood of expected distribution given actual observation
    """
    freq_expected = get_corrected_multinomial_frequencies(expected)
    return multinomial.pmf(x=actual, n=sum(actual), p=freq_expected)


def multinomial_likelihood_ratio(actual: List[int], expected: List[int]) -> Tuple[float, float]:
    likelihood = multinomial_likelihood(actual, expected)
    max_likelihood = multinomial_likelihood(actual, actual)
    likelihood_ratio = likelihood / max_likelihood
    return likelihood, likelihood_ratio


# Metrics functions


def get_precision(fp: int, tp: int) -> float:
    """
    Get the precision, as defined from tp and fp

    Parameters
    ----------
    fp : int
        number of false positive observation (detected but false)
    tp : int
        number of true positive observations (detected true)

    Returns
    -------
    The precision score
    """
    return 1 - safe_divide(fp, fp + tp)


def get_recall(fn: int, tp: int) -> float:
    """
    Get the recall, as defined from tp and fn

    Parameters
    ----------
    fn : int
        number of false negative observations (missed true)
    tp : int
        number of true positive observation (detected true)

    Returns
    -------
    The recall score
    """

    return 1 - safe_divide(fn, fn + tp)


def get_f1(precision: float, recall: float) -> float:
    """
    Get the F1 score (harmonic mean of precision and recall)

    Parameters
    ----------
    precision : int
        precision of the experiment
    recall : int
        recall of the experiment

    Returns
    -------
    The F1 score of the experiment
    """
    return safe_divide(2 * precision * recall, precision + recall)
