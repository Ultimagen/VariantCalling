import numpy as np
import pathmagic
import pandas as pd
import pickle
from os.path import join as pjoin
from pathmagic import PYTHON_TESTS_PATH
from python.pipelines import convert_h5_to_json_utils as convert
import re

def test_preprocess_h5_key_with_slash():
    assert convert.preprocess_h5_key("/foo") == "foo"

def test_preprocess_h5_key_without_slash():
    assert convert.preprocess_h5_key("foo") == "foo"

def test_should_skip_h5_key_true():
    assert convert.should_skip_h5_key("str_histogram_123str", "histogram")

def test_should_skip_h5_key_false():
    assert not convert.should_skip_h5_key("str_his_togram_123str", "histogram")

def test_preprocess_json_for_mongodb():
    metrics_h5_path = pjoin(PYTHON_TESTS_PATH, "h5_to_json/140479-BC21_aggregated_metrics.h5")
    json_str = convert.convert_h5_to_json(metrics_h5_path, "metrics", True, "histogram")
    assert re.search(r"%", json_str) is None

def test_do_not_preprocess_json_for_mongodb():
    metrics_h5_path = pjoin(PYTHON_TESTS_PATH, "h5_to_json/140479-BC21_aggregated_metrics.h5")
    json_str = convert.convert_h5_to_json(metrics_h5_path, "metrics", False, "histogram")
    assert re.search(r"%", json_str) is not None

def test_get_h5_keys():
    metrics_h5_path = pjoin(PYTHON_TESTS_PATH, "h5_to_json/140479-BC21_aggregated_metrics.h5")
    assert np.array_equal(convert.get_h5_keys(metrics_h5_path), ['/AlignmentSummaryMetrics', '/DuplicationMetrics', '/GcBiasDetailMetrics', '/GcBiasSummaryMetrics', '/QualityYieldMetrics', '/RawWgsMetrics', '/WgsMetrics', '/histogram_AlignmentSummaryMetrics', '/histogram_RawWgsMetrics', '/histogram_WgsMetrics', '/histogram_coverage', '/stats_coverage'])

def test_convert_h5_to_json():
    metrics_h5_path = pjoin(PYTHON_TESTS_PATH, "h5_to_json/140479-BC21_aggregated_metrics.h5")
    metrics_json_path = pjoin(PYTHON_TESTS_PATH, "h5_to_json/140479-BC21_aggregated_metrics.json")
    with open(metrics_json_path, 'r') as json_file:
        data = json_file.read()
    assert convert.convert_h5_to_json(metrics_h5_path, "metrics", True, "histogram") == data
