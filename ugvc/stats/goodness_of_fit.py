from typing import List, Tuple

import numpy as np
from scipy.stats import multinomial


def scale_contingency_table(table: List[int], n: int) -> List[int]:
    sum_table = sum(table)
    if sum_table > 0:
        scaled_table = np.array(table) * (n / sum_table)
        return list(np.round(scaled_table).astype(int))
    else:
        return table


def get_corrected_multinomial_frequencies(counts: List[int]) -> np.array:
    corrected_counts = (np.array(counts) + 1) / max(sum(counts), 1)
    return corrected_counts / np.sum(corrected_counts)


def multinomial_likelihood(actual: List[int], expected: List[int]) -> float:
    freq_expected = get_corrected_multinomial_frequencies(expected)
    return multinomial.pmf(actual, sum(actual), freq_expected)


def multinomial_likelihood_ratio(actual: List[int], expected: List[int]) -> Tuple[float, float]:
    likelihood = multinomial_likelihood(actual, expected)
    max_likelihood = multinomial_likelihood(actual, actual)
    likelihood_ratio = likelihood / max_likelihood
    return likelihood, likelihood_ratio
