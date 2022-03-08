import unittest

import pandas as pd

from python.pipelines.variant_filtering_utils import SingleTrivialClassifierModel


class TestVariantFilteringModels(unittest.TestCase):

    def test_single_trivial_classifier(self):
        model = SingleTrivialClassifierModel()
        df = pd.DataFrame({'filter': ['PASS', 'HPOL_RUN', 'LOW_SCORE', 'COHORT_FP;HPOL_RUN']})

        predictions = model.predict(df)
        self.assertEqual(['tp', 'fp', 'fp', 'fp'], list(predictions))

        model = SingleTrivialClassifierModel(ignored_filters=['HPOL_RUN'])
        predictions = model.predict(df)
        self.assertEqual(['tp', 'tp', 'fp', 'fp'], list(predictions))

        model = SingleTrivialClassifierModel(ignored_filters=['HPOL_RUN', 'COHORT_FP'])
        predictions = model.predict(df)
        self.assertEqual(['tp', 'tp', 'fp', 'tp'], list(predictions))



