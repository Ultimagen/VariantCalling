from pathlib import Path
from test import get_resource_dir

import papermill
import pytest
from papermill.exceptions import PapermillExecutionError

from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import (
    TEMPLATE_NOTEBOOK,
    Inputs,
    Thresholds,
)
from ugvc.pipelines.single_cell_qc.single_cell_qc_pipeline import (
    generate_report,
    prepare_parameters_for_report,
    single_cell_qc,
)

inputs_dir = Path(get_resource_dir(__file__)) # returns VariantCalling/test/resources/unit/single_cell_qc + the current file name: test_single_cell_qc_pipeline
inputs_dir = inputs_dir.parent # the resources are in the directory: VariantCalling/test/resources/unit/single_cell_qc so need to drop the last part of the path

inputs = Inputs(
    trimmer_stats_csv=str(Path(inputs_dir) / "trimmer_stats.csv"),
    trimmer_histogram_csv=[str(Path(inputs_dir) / "trimmer_histogram.csv")],
    trimmer_failure_codes_csv=str(Path(inputs_dir) / "trimmer_failure_codes.csv"),
    sorter_stats_csv=str(Path(inputs_dir) / "sorter_stats.csv"),
    star_stats=str(Path(inputs_dir) / "star_insert_Log.final.out"),
    star_reads_per_gene=str(Path(inputs_dir) / "star_insert_ReadsPerGene.out.tab"),
    insert=str(Path(inputs_dir) / "insert_subsample.fastq.gz"),
)
thresholds = Thresholds(
    pass_trim_rate=0.9,
    read_length=100,
    fraction_below_read_length=0.1,
    percent_aligned=0.9,
)
sample_name = "test_sample"
h5_file = Path(inputs_dir) / "single_cell_qc_stats.h5"


def test_single_cell_qc(tmpdir):
    tmpdir = Path(tmpdir)
    single_cell_qc(
        input_files=inputs,
        output_path=tmpdir,
        thresholds=thresholds,
        sample_name=sample_name,
    )

    # assert file with extensions .h5 exists
    assert any(file.suffix == ".h5" for file in tmpdir.iterdir())
    # assert file with extensions .html exists
    assert any(file.suffix == ".html" for file in tmpdir.iterdir())
    # assert no other files exist
    assert len(list(tmpdir.iterdir())) == 2


def test_prepare_parameters_for_report(tmpdir):
    tmpdir = Path(tmpdir)
    parameters, tmp_files = prepare_parameters_for_report(h5_file, thresholds, tmpdir)

    # assert parameters is not empty
    assert parameters
    # assert tmp_files is not empty
    assert tmp_files
    # assert all files in tmp_files exist
    assert all(file.exists() for file in tmp_files)
    # assert parameters keys are equal to expected notebook_parameters keys
    notebook_parameters = papermill.inspect_notebook(str(TEMPLATE_NOTEBOOK))
    assert set(parameters.keys()) == set(notebook_parameters.keys())


def test_prepare_parameters_for_report_h5_not_found():
    h5_file = Path(inputs_dir) / "non_existent_file.h5"
    # assert exception is raised when h5 file is not found
    with pytest.raises(FileNotFoundError):
        parameters, tmp_files = prepare_parameters_for_report(
            h5_file, thresholds, "tmpdir"
        )


def test_prepare_parameters_for_report_outpath_not_exist():
    output_path = "non_existent_dir"
    # assert exception is raised when outpath does not exist
    with pytest.raises(FileNotFoundError):
        parameters, tmp_files = prepare_parameters_for_report(
            h5_file, thresholds, output_path
        )


def test_generate_report(tmpdir):
    tmpdir = Path(tmpdir)
    parameters, tmp_files = prepare_parameters_for_report(h5_file, thresholds, tmpdir)
    report_html = generate_report(parameters, tmpdir, tmp_files, sample_name)

    # assert report_html exists
    assert report_html.exists()

    # assert report_html is not empty
    assert report_html.stat().st_size > 0

    # assert no other files exist
    assert len(list(tmpdir.iterdir())) == 1


def test_generate_report_tmpfile_not_a_file(tmpdir):
    tmpdir = Path(tmpdir)
    parameters, tmp_files = prepare_parameters_for_report(h5_file, thresholds, tmpdir)
    tmp_files.append(tmpdir)
    generate_report(parameters, tmpdir, tmp_files, sample_name)


def test_generate_report_missing_parameter(tmpdir):
    tmpdir = Path(tmpdir)
    with pytest.raises(PapermillExecutionError):
        generate_report({}, tmpdir, [], sample_name)
