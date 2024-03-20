from __future__ import annotations

import numpy as np
from scipy.stats import multinomial
from sklearn import metrics

from ugvc.utils.math_utils import safe_divide

# Goodness of fit functions


def scale_contingency_table(table: list[int], n: int) -> list[int]:
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
    # if sum_table == 0:
    return table


def correct_multinomial_frequencies(counts: list[int]) -> np.ndarray:
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


def multinomial_likelihood(actual: list[int], expected: list[int]) -> float:
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
    freq_expected = correct_multinomial_frequencies(expected)
    return multinomial.pmf(x=actual, n=sum(actual), p=freq_expected)


def multinomial_likelihood_ratio(actual: list[int], expected: list[int]) -> tuple[float, float]:
    likelihood = multinomial_likelihood(actual, expected)
    max_likelihood = multinomial_likelihood(actual, actual)
    likelihood_ratio = likelihood / max_likelihood
    return likelihood, likelihood_ratio


# Metrics functions


def get_precision(false_positives: int, true_positives: int, return_if_denominator_is_0=1) -> float:
    """
    Get the precision, as defined from tp and false_positives

    Parameters
    ----------
    false_positives : int
        number of false positive observation (detected but false)
    true_positives : int
        number of true positive observations (detected true)
    return_if_denominator_is_0: any
        return value if tp + fp == 0
    Returns
    -------
    The precision score
    """
    if false_positives + true_positives == 0:
        return return_if_denominator_is_0
    return 1 - false_positives / (false_positives + true_positives)


def get_recall(false_negatives: int, true_positives: int, return_if_denominator_is_0=1) -> float:
    """
    Get the recall, as defined from true_positives and false_negatives

    Parameters
    ----------
    false_negatives : int
        number of false negative observations (missed true)
    true_positives : int
        number of true positive observation (detected true)
    return_if_denominator_is_0: any
        return value if tp + fn == 0

    Returns
    -------
    The recall score
    """
    if false_negatives + true_positives == 0:
        return return_if_denominator_is_0
    return 1 - false_negatives / (false_negatives + true_positives)


def get_f1(precision: float, recall, null_value=np.nan) -> float:
    """
    Get the F1 score (harmonic mean of precision and recall)

    Parameters
    ----------
    precision : float
        precision of the experiment
    recall : float
        recall of the experiment
    null_value: Any
        return this null_value if recall or precision are equal to this null_value

    Returns
    -------
    The F1 score of the experiment
    """
    if null_value in {precision, recall}:
        return null_value
    return safe_divide(2 * precision * recall, precision + recall)


def precision_recall_curve(
    gtr: np.ndarray,
    predictions: np.ndarray,
    fn_mask: np.ndarray,
    pos_label: str | int | None = 1,
    min_class_counts_to_output: int = 20,
) -> tuple:
    """Calculates precision/recall curve from double prediction scores and ground truth
    Similar to sklearn precision_recall curve, but uses mask of variants that were false negatives

    Parameters
    ----------
    gtr: np.ndarray
        String array of ground truth labels. FN should be labeled 'tp'
    predictions: np.ndarray
        Array of prediction scores
    fn_mask: np.ndarray
        Boolean array that has true on the locations that were false negatives
    pos_label: str or 0
        Label of true call, otherwise 1s will be considered
    min_class_counts_to_output: int
        Limit on the count of classes (denominator) below which the results are too noisy and not calculated

    Returns
    -------
    tuple
        precisions, recalls, f1_score, prediction_values
    """

    if len(gtr) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    assert len(set(gtr)) <= 2, "Only up to two classes of variant labels are possible"
    assert len(fn_mask) == len(predictions), "FN mask should be of the length of predictions"

    gtr_select = gtr[~fn_mask]
    gtr_select = gtr_select == pos_label
    predictions_select = predictions[~fn_mask]
    original_fn_count = fn_mask.sum()

    if len(gtr_select) > 0:
        raw_precision, raw_recall, thresholds = metrics.precision_recall_curve(
            gtr_select, predictions_select, pos_label=True
        )
    else:
        raw_precision = np.array([gtr_select.sum() / len(gtr_select), 1.0])
        raw_recall = np.array([1.0, 0.0])
        thresholds = np.array([np.min(predictions_select)])

    recall_correction = gtr_select.sum() / (gtr_select.sum() + original_fn_count)
    recalls = raw_recall * recall_correction
    recalls = recalls[
        1:-1
    ]  # remove the 1,0 value that sklearn adds, remove the initial value that the precision_recall_curve adds
    precisions = raw_precision[1:-1]
    thresholds = thresholds[1:]
    f1_score = 2 * (recalls * precisions) / (recalls + precisions + np.finfo(float).eps)

    # Find the score cutoff at which too few calls remain (to remove areas where the precision recall curve is noisy)
    predictions_select = np.sort(predictions_select)
    if len(predictions_select) > 0:
        threshold_cutoff = predictions_select[max(0, len(predictions_select) - min_class_counts_to_output)]
    else:
        threshold_cutoff = 0

    mask = thresholds > threshold_cutoff
    return precisions[~mask], recalls[~mask], f1_score[~mask], thresholds[~mask]


def generate_sample_from_dist(vals: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Returns values from a distribution

    Parameters
    ----------
    vals: np.ndarray
        Values
    probs: np.ndarray
        Probabilities

    Returns
    -------
    np.ndarray
    """
    return np.random.choice(vals, 10000, p=probs)
