from typing import List, Tuple, Optional, Union

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


def precision_recall_curve(gtr: np.ndarray, predictions: np.ndarray,
                           pos_label: Optional[Union[str, int]] = 1,
                           fn_score: float = -1,
                           min_class_counts_to_output: int = 20) -> tuple:
    '''Calculates precision/recall curve from double prediction scores and gtr

    Parameters
    ----------
    gtr: np.ndarray
        String array of ground truth
    predictions: np.ndarray
        Array of prediction scores
    pos_label: str or 0
        Label of true call, otherwise 1s will be considered
    fn_score: float
        Score of false negative. Should be such that they are the lowest scoring variants
    min_class_counts_to_output: int
        Limit on the count of classes (denominator) below which the results are too noisy and not calculated

    Returns
    -------
    tuple
        precisions, recalls, prediction_values
    '''
    asidx = np.argsort(predictions)
    predictions = predictions[asidx]
    gtr = gtr[asidx]
    gtr = gtr == pos_label

    tp_counts = np.cumsum(gtr[::-1])[::-1]
    fn_counts = np.cumsum(predictions == fn_score)
    fp_counts = np.cumsum((gtr == 0)[::-1])[::-1]
    mask = (tp_counts + fp_counts < min_class_counts_to_output) |\
           (tp_counts + fn_counts < min_class_counts_to_output)
    precisions = tp_counts / (tp_counts + fp_counts)
    recalls = tp_counts / (tp_counts + fn_counts)
    f1 = 2 * (recalls * precisions) / \
        (recalls + precisions + np.finfo(float).eps)

    trim_idx = np.argmax(predictions > fn_score)
    mask = mask[trim_idx:]
    return precisions[trim_idx:][~mask], recalls[trim_idx:][~mask], \
        f1[trim_idx:][~mask], predictions[trim_idx:][~mask]
