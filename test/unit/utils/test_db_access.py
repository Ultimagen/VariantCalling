import re
from os.path import join as pjoin
from test import get_resource_dir

import pandas as pd
import pytest

import ugvc.utils.db_access as db_access


class TestDBAccess:
    input_dir = get_resource_dir(__file__)

    def test_database_access_available(self):
        assert (
            not db_access.DISABLE_PAPYRUS_ACCESS
        ), "Papyrus command line access not available, need to install pymongo and define $PAPYRUS_ACCESS_STRING"

    def test_fetch_from_database(self):
        r = re.compile(r"150450.*")
        docs = db_access.query_database(
            {"metadata.runId": r, "inputs": {"$exists": True}}
        )
        assert len(docs) >= 8, "Was not able to fetch at least 8 documents"

        db_access.DISABLE_PAPYRUS_ACCESS = True
        with pytest.raises(AssertionError, match=r"Database access.*"):
            docs = db_access.query_database(
                {"metadata.runId": r, "inputs": {"$exists": True}}
            )
        db_access.DISABLE_PAPYRUS_ACCESS = False

    hardcoded_wfids = [
        "de06922f-07f8-4b51-843e-972308c81c6f",
        "ea5e54d8-3db1-47b2-bc0e-68f38e3e89f3",
        "9588412a-e8c2-447d-b1e7-dc46b5da3fb4",
        "469b7436-a737-4257-9f08-7990ff95a461",
        "0440f417-be57-4887-b668-39c47cbd55aa",
        "2989cdc2-6fa5-4931-adcf-a9c7372f162a",
        "ddc419d2-5b94-4b92-85e0-6f17c56f3e4d",
        "6674c3cc-410d-47ec-9718-b23fa34c86e1",
        "6289c2fa-0c6a-4de8-805e-cae0e227227f",
        "1e4440f9-76a3-4bd0-8359-8c653d5f7212",
    ]

    def test_inputs_outputs_dataframe(self):
        r = re.compile(r"150450.*")
        docs = db_access.query_database(
            {"metadata.runId": r, "inputs": {"$exists": True}}
        )
        docs = sorted(docs, key=lambda x: x["metadata"]["workflowId"])
        docs = [x for x in docs if x["metadata"]["workflowId"] in self.hardcoded_wfids]
        all_inputs = pd.concat((db_access.inputs2df(x) for x in docs), axis=0)
        assert all_inputs.equals(
            pd.read_hdf(pjoin(self.input_dir, "expected_inputs_df.h5"), key="df")
        )

    def test_metrics_dataframe(self):
        r = re.compile(r"150450.*")
        docs = db_access.query_database(
            {"metadata.runId": r, "inputs": {"$exists": True}}
        )
        docs = sorted(docs, key=lambda x: x["metadata"]["workflowId"])
        docs = [x for x in docs if x["metadata"]["workflowId"] in self.hardcoded_wfids]

        metrics_to_report = [
            "AlignmentSummaryMetrics",
            "Contamination",
            "DuplicationMetrics",
            "GcBiasDetailMetrics",
            "GcBiasSummaryMetrics",
            "QualityYieldMetrics",
            "RawWgsMetrics",
            "WgsMetrics",
            "stats_coverage",
            "short_report_/all_data",
            "short_report_/all_data_gt",
        ]

        all_inputs = pd.concat(
            (db_access.metrics2df(x, metrics_to_report) for x in docs), axis=0
        )

        assert all_inputs.equals(
            pd.read_hdf(pjoin(self.input_dir, "expected_metrics_df.h5"), key="df")
        )
