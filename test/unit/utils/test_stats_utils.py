import unittest

import numpy as np

from ugvc.utils.stats_utils import (
    correct_multinomial_frequencies,
    get_f1,
    get_precision,
    get_recall,
    multinomial_likelihood,
    multinomial_likelihood_ratio,
    precision_recall_curve,
    scale_contingency_table,
)


class TestStatsUtils(unittest.TestCase):
    def test_scale_up_uniform_contingency_table(self):
        table = [1, 1, 1]
        self.assertEqual([1, 1, 1], scale_contingency_table(table, 2))
        self.assertEqual([1, 1, 1], scale_contingency_table(table, 3))
        self.assertEqual([1, 1, 1], scale_contingency_table(table, 4))
        self.assertEqual([2, 2, 2], scale_contingency_table(table, 5))
        self.assertEqual([2, 2, 2], scale_contingency_table(table, 6))
        self.assertEqual([2, 2, 2], scale_contingency_table(table, 7))
        self.assertEqual([3, 3, 3], scale_contingency_table(table, 9))

    def test_scale_down_uniform_contingency_table(self):
        table = [10, 10, 10]
        self.assertEqual([1, 1, 1], scale_contingency_table(table, 2))
        self.assertEqual([1, 1, 1], scale_contingency_table(table, 3))
        self.assertEqual([1, 1, 1], scale_contingency_table(table, 4))
        self.assertEqual([2, 2, 2], scale_contingency_table(table, 5))
        self.assertEqual([2, 2, 2], scale_contingency_table(table, 6))
        self.assertEqual([2, 2, 2], scale_contingency_table(table, 7))
        self.assertEqual([3, 3, 3], scale_contingency_table(table, 9))

    def test_scale_non_uniform_contingency_table(self):
        table = [10, 20, 25]
        self.assertEqual([18, 36, 45], scale_contingency_table(table, 100))
        self.assertEqual([2, 4, 5], scale_contingency_table(table, 10))

    def test_correct_multinomial_frequencies(self):
        np.testing.assert_array_equal(np.array([1, 1, 1]) / 3, correct_multinomial_frequencies([10, 10, 10]))
        np.testing.assert_array_equal(
            np.array([11, 11, 1]) / 23,
            correct_multinomial_frequencies([10, 10, 0]),
        )

    def test_multinomial_likelihood(self):
        self.assertAlmostEqual(0.0652, multinomial_likelihood([4, 4, 4], [4, 4, 4]), places=3)
        self.assertAlmostEqual(0.0652, multinomial_likelihood([4, 4, 4], [40, 40, 40]), places=3)
        self.assertAlmostEqual(0.0068, multinomial_likelihood([40, 40, 40], [40, 40, 40]), places=3)

        self.assertAlmostEqual(3.3 * 10**-13, multinomial_likelihood([4, 4, 40], [4, 4, 4]), places=10)
        self.assertAlmostEqual(3.3 * 10**-13, multinomial_likelihood([4, 4, 40], [40, 40, 40]), places=10)

        # Get expected improvement as actual becomes fitter to expected
        self.assertAlmostEqual(
            2.1 * 10**-10,
            multinomial_likelihood([10, 10, 10], [1, 10, 40]),
            places=10,
        )
        self.assertAlmostEqual(2.7 * 10**-53, multinomial_likelihood([40, 10, 1], [1, 10, 40]), places=40)
        self.assertAlmostEqual(0.039, multinomial_likelihood([1, 10, 40], [1, 10, 40]), places=3)

        # 0 expected does not cause 0 likelihood, because of add-one correction
        self.assertAlmostEqual(0.0043, multinomial_likelihood([4, 4, 4], [4, 4, 0]), places=3)

        # need to watch out from this edge-case
        self.assertAlmostEqual(3.3 * 10**-13, multinomial_likelihood([4, 4, 40], [0, 0, 0]), places=3)

    def test_multinomial_likelihood_ratio(self):
        self.assertAlmostEqual(1, multinomial_likelihood_ratio([4, 4, 4], [4, 4, 4])[1], places=3)
        self.assertAlmostEqual(1, multinomial_likelihood_ratio([4, 4, 4], [40, 40, 40])[1], places=3)
        self.assertAlmostEqual(1, multinomial_likelihood_ratio([40, 40, 40], [40, 40, 40])[1], places=3)

        self.assertAlmostEqual(
            3.3 * 10**-13,
            multinomial_likelihood_ratio([4, 4, 40], [4, 4, 4])[1],
            places=10,
        )
        self.assertAlmostEqual(
            3.3 * 10**-13,
            multinomial_likelihood_ratio([4, 4, 40], [40, 40, 40])[1],
            places=10,
        )

        # Get expected improvement as actual becomes fitter to expected
        self.assertAlmostEqual(
            7.8 * 10**-9,
            multinomial_likelihood_ratio([10, 10, 10], [1, 10, 40])[1],
            places=10,
        )
        self.assertAlmostEqual(
            6.9 * 10**-52,
            multinomial_likelihood_ratio([40, 10, 1], [1, 10, 40])[1],
            places=40,
        )
        self.assertAlmostEqual(1, multinomial_likelihood_ratio([1, 10, 40], [1, 10, 40])[1], places=3)

        # 0 expected does not cause 0 likelihood, because of add-one correction
        self.assertAlmostEqual(0.0661, multinomial_likelihood_ratio([4, 4, 4], [4, 4, 0])[1], places=3)

        # need to watch out from this edge-case
        self.assertAlmostEqual(
            9.1 * 10**-12,
            multinomial_likelihood_ratio([4, 4, 40], [0, 0, 0])[1],
            places=10,
        )

    def test_get_precision(self):
        self.assertAlmostEqual(get_precision(100, 900), 0.9)
        self.assertAlmostEqual(get_precision(1, 900), 0.99889, places=5)

    def test_get_recall(self):
        self.assertAlmostEqual(0.9, get_recall(100, 900))
        self.assertAlmostEqual(0.99889, get_recall(1, 900), places=5)

    def test_get_f1(self):
        self.assertAlmostEqual(0.942857, get_f1(recall=0.99, precision=0.9), places=5)
        self.assertAlmostEqual(0.642857, get_f1(recall=0.5, precision=0.9), places=5)

    def test_precision_recall_curve(self):
        labels = np.array([0, 1] * 50)
        scores = np.array([0.1, 0.8] * 50)
        precision, recalls, f1, predictions = precision_recall_curve(
            labels,
            scores,
            fn_mask=np.zeros_like(scores, dtype=bool),
            pos_label=1,
            min_class_counts_to_output=1,
        )
        self.assertEqual(len(precision), 1)
        self.assertEqual(len(f1), 1)
        self.assertAlmostEqual(max(f1), 1)

        labels = np.array([0, 1] * 50 + [1] * 10)
        scores = np.array([0.1, 0.8] * 50 + [-1] * 10)
        precision, recalls, f1, predictions = precision_recall_curve(
            labels,
            scores,
            np.concatenate((np.zeros(100, dtype=bool), np.ones(10, dtype=bool))),
            pos_label=1,
            min_class_counts_to_output=1,
        )
        self.assertEqual(len(precision), 1)
        self.assertEqual(len(f1), 1)
        self.assertAlmostEqual(max(f1), 0.909090909)

        labels = []
        scores = []
        precision, recalls, f1, predictions = precision_recall_curve(
            labels, scores, np.array([]), pos_label=1, min_class_counts_to_output=1
        )
        self.assertEqual(len(precision), 0)
        self.assertEqual(len(f1), 0)
