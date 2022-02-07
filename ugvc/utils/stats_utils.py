from typing import List, Tuple

import numpy as np
from scipy.stats import multinomial

from ugvc.utils.math_utils import safe_divide


# Goodness of fit functions


def scale_contingency_table(table: List[int], n: int) -> List[int]:
    """
    @param table: contingency table represented as list
    @param n: scale-factor
    @return: a contingency table where the sum of categories is approximately n
    """
    sum_table = sum(table)
    if sum_table > 0:
        scaled_table = np.array(table) * (n / sum_table)
        return list(np.round(scaled_table).astype(int))
    else:
        return table


def get_corrected_multinomial_frequencies(counts: List[int]) -> np.array:
    """
    @param counts: contingency table represented as list of ints
    @return: frequency of each category after add-one correction
    """
    corrected_counts = np.array(counts) + 1
    return corrected_counts / np.sum(corrected_counts)


def multinomial_likelihood(actual: List[int], expected: List[int]) -> float:
    """
    Calculate the likelihood of observing actual, under the multinomial distribution described by expected
    @param actual: a list of integer observations
    @param expected: a list integers from which to for which to fit a multinomial distribution to
    @return: likelihood
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
    @param fp: number of false positives
    @param tp:  number of true positives
    @return: precision
    """
    return 1 - safe_divide(fp, fp + tp)


def get_recall(fn: int, tp: int) -> float:
    """
    @param fn: number of false negatives
    @param tp: number of true positives
    @return: recall
    """
    return 1 - safe_divide(fn,  fn + tp)


def get_f1(precision: float, recall: float) -> float:
    """
    @param precision: the precision
    @param recall: the recall
    @return: f1 score
    """
    return safe_divide(2 * precision * recall, precision + recall)