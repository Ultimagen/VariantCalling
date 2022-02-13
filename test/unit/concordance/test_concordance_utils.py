import unittest

from ugvc.concordance.concordance_utils import *


class TestConcordanceUtils(unittest.TestCase):

    def test_calc_accuracy_metrics(self):
        """
        given concordance dataframe with all rows passing filter, gather expected metrics for expected variant type
        """
        concordance_df = pd.DataFrame({'classify': ['tp', 'tp', 'fp', 'fn', 'tp', 'tn'],
                                       'filter': ['PASS'] * 6,
                                       'tree_score': [1] * 6,
                                       'hmer_indel_nuc': ['N'] * 6,
                                       'indel': [True] * 6,
                                       'hmer_indel_length': [2] * 6
                                       })
        accuracy_df = calc_accuracy_metrics(concordance_df, 'classify')
        expected = {'initial_tp': [3],
                    'initial_fp': [1],
                    'initial_fn': [1],
                    'initial_precision': [0.75],
                    'initial_recall': [0.75],
                    'initial_f1': [0.75],
                    'tp': [3],
                    'fp': [1],
                    'fn': [1],
                    'precision': [0.75],
                    'recall': [0.75],
                    'f1': [0.75]
                    }
        # DataFrame dict contains index->value dictionaries per each column
        expected_indels = {'group': {7: 'INDELS'}}
        expected_hmer_indel_lt_4 = {'group': {2: 'HMER indel <= 4'}}
        for expected_key, expected_value in expected.items():
            expected_hmer_indel_lt_4[expected_key] = {2: expected_value[0]}
            expected_indels[expected_key] = {7: expected_value[0]}

        self.assertEqual(expected_hmer_indel_lt_4, accuracy_df[accuracy_df['group'] == 'HMER indel <= 4'].to_dict())
        self.assertEqual(expected_indels, accuracy_df[accuracy_df['group'] == 'INDELS'].to_dict())

    def test_calc_accuracy_metrics_with_non_passing_rows(self):
        """
        given concordance dataframe with some rows failing filter, gather expected metrics for expected variant type
        filtered fp should raise precision
        filtered fn should have no effect
        """
        concordance_df = pd.DataFrame({'classify': ['tp', 'tp', 'fp', 'fn', 'tp', 'tn'],
                                       'filter': ['PASS', 'PASS', 'SEC', 'LOW_SCORE', 'PASS', 'PASS'],
                                       'tree_score': [1] * 6,
                                       'hmer_indel_nuc': ['N'] * 6,
                                       'indel': [False] * 6,
                                       'hmer_indel_length': [None] * 6
                                       })
        accuracy_df = calc_accuracy_metrics(concordance_df, 'classify')
        expected = {'initial_tp': [3],
                    'initial_fp': [1],
                    'initial_fn': [1],
                    'initial_precision': [0.75],
                    'initial_recall': [0.75],
                    'initial_f1': [0.75],
                    'tp': [3],
                    'fp': [0],
                    'fn': [1],
                    'precision': [1],
                    'recall': [0.75],
                    'f1': [0.85714]
                    }

        # DataFrame dict contains index->value dictionaries per each column
        expected_snps = {'group': {0: 'SNP'}}

        for expected_key, expected_value in expected.items():
            expected_snps[expected_key] = {0: expected_value[0]}

        self.assertEqual(expected_snps, accuracy_df[accuracy_df['group'] == 'SNP'].to_dict())

    def test_calc_recall_precision_curve(self):
        """
        given concordance dataframe with all rows passing filter, calc recall/precision curve
        """
        concordance_df = pd.DataFrame({'classify': ['tp', 'tp', 'fp', 'fn', 'tp', 'tn'],
                                       'filter': ['PASS'] * 6,
                                       'tree_score': [1] * 6,
                                       'hmer_indel_nuc': ['N'] * 6,
                                       'indel': [True] * 6,
                                       'hmer_indel_length': [2] * 6
                                       })
        accuracy_df = calc_recall_precision_curve(concordance_df, 'classify')
        expected = {'initial_tp': [3],
                    'initial_fp': [1],
                    'initial_fn': [1],
                    'initial_precision': [0.75],
                    'initial_recall': [0.75],
                    'initial_f1': [0.75],
                    'tp': [3],
                    'fp': [1],
                    'fn': [1],
                    'precision': [0.75],
                    'recall': [0.75],
                    'f1': [0.75]
                    }
        # DataFrame dict contains index->value dictionaries per each column
        expected_indels = {'group': {7: 'INDELS'}}
        expected_hmer_indel_lt_4 = {'group': {2: 'HMER indel <= 4'}}
        for expected_key, expected_value in expected.items():
            expected_hmer_indel_lt_4[expected_key] = {2: expected_value[0]}
            expected_indels[expected_key] = {7: expected_value[0]}

        self.assertEqual(expected_hmer_indel_lt_4, accuracy_df[accuracy_df['group'] == 'HMER indel <= 4'].to_dict())
        self.assertEqual(expected_indels, accuracy_df[accuracy_df['group'] == 'INDELS'].to_dict())
