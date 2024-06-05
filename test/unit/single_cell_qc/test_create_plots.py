from pathlib import Path
from test import get_resource_dir
from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt

from ugvc.pipelines.single_cell_qc import create_plots
from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import OutputFiles


@pytest.fixture
def inputs_dir():
    # returns VariantCalling/test/resources/unit/single_cell_qc + the current file name: test_single_cell_qc_pipeline
    inputs_dir = Path(get_resource_dir(__file__))
    # the resources are in the directory: VariantCalling/test/resources/unit/single_cell_qc so need to drop the last part of the path
    inputs_dir = inputs_dir.parent
    return inputs_dir


@pytest.fixture
def h5_file(inputs_dir):
    return inputs_dir / "single_cell_qc_stats.h5"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir)


@patch("ugvc.pipelines.single_cell_qc.create_plots.set_pyplot_defaults")
def test_cbc_umi_plot(mock_set_pyplot_defaults, h5_file, output_path):
    result = create_plots.cbc_umi_plot(h5_file, output_path)
    expected_output = output_path / OutputFiles.CBC_UMI_PLOT.value
    assert result == expected_output
    # check that the plot contains data
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(plt.imread(result))
    assert ax.has_data()


@patch("ugvc.pipelines.single_cell_qc.create_plots.set_pyplot_defaults")
def test_plot_insert_length_histogram(mock_set_pyplot_defaults, h5_file, output_path):
    result = create_plots.plot_insert_length_histogram(h5_file, output_path)
    expected_output = output_path / OutputFiles.INSERT_LENGTH_HISTOGRAM.value
    assert result == expected_output
    # check that the plot contains data
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(plt.imread(result))
    assert ax.has_data()


@patch("ugvc.pipelines.single_cell_qc.create_plots.set_pyplot_defaults")
def test_plot_mean_insert_quality_histogram(
    mock_set_pyplot_defaults, h5_file, output_path
):
    result = create_plots.plot_mean_insert_quality_histogram(h5_file, output_path)
    expected_output = output_path / OutputFiles.MEAN_INSERT_QUALITY_PLOT.value
    assert result == expected_output
    # check that the plot contains data
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(plt.imread(result))
    assert ax.has_data()


@patch("ugvc.pipelines.single_cell_qc.create_plots.set_pyplot_defaults")
def test_plot_quality_per_position(
    mock_set_pyplot_defaults,
    h5_file,
    output_path,
):
    result = create_plots.plot_quality_per_position(h5_file, output_path)
    expected_output = output_path / OutputFiles.QUALITY_PER_POSITION_PLOT.value
    assert result == expected_output
    # check that the plot contains data
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(plt.imread(result))
    assert ax.has_data()
