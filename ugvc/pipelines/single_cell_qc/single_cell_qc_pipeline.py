from argparse import ArgumentParser
from pathlib import Path

import nbformat
import pandas as pd
import papermill
from nbconvert import HTMLExporter

from ugvc.pipelines.single_cell_qc.collect_statistics import (
    collect_statistics,
    extract_statistics_table,
)
from ugvc.pipelines.single_cell_qc.create_plots import (
    cbc_umi_plot,
    plot_insert_length_histogram,
    plot_mean_insert_quality_histogram,
    plot_quality_per_position,
)
from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import (
    TEMPLATE_NOTEBOOK,
    H5Keys,
    Inputs,
    OutputFiles,
    Thresholds,
)
from ugvc.utils.metrics_utils import convert_h5_to_json
from ugvc.utils.misc_utils import modify_jupyter_notebook_html


def single_cell_qc(
    input_files: Inputs, output_path: str, thresholds: Thresholds, sample_name: str
):
    """
    Run single cell qc pipeline that collects statistics, prepares parameters for report and generates report

    Parameters
    ----------
    input_files : Inputs
        Inputs object with paths to input files
    output_path : str
        Path to output directory
    thresholds : Thresholds
        Thresholds object with thresholds for qc
    sample_name : str
        Sample name to be included as a prefix in the output files
    """
    if not sample_name.endswith("."):
        sample_name += "."

    h5_file = collect_statistics(input_files, output_path, sample_name)
    extract_statistics_table(h5_file)

    params, tmp_files = prepare_parameters_for_report(h5_file, thresholds, output_path)
    output_report_html = generate_report(params, output_path, tmp_files, sample_name)

    # keep only STAR and short table data in h5 file
    with pd.HDFStore(h5_file, "a") as store:
        keys_to_keep = [
            H5Keys.STATISTICS_SHORTLIST.value,
            H5Keys.STAR_STATS.value,
            H5Keys.STAR_READS_PER_GENE.value,
        ]
        for key in store.keys():
            if key.strip('/') not in keys_to_keep:
                store.remove(key)
    
    # convert h5 file to json
    statistics_json_file = Path(output_path) / (sample_name + OutputFiles.STATISTICS_JSON.value)
    convert_h5_to_json(
        input_h5_filename=h5_file,
        root_element="metrics",
        ignored_h5_key_substring=H5Keys.STAR_READS_PER_GENE.value,
        output_json=statistics_json_file
    )

def prepare_parameters_for_report(
    h5_file: Path, thresholds: Thresholds, output_path: str
) -> tuple[dict, list[Path]]:
    """
    Prepare parameters for report generation (h5 file, thresholds, plots)

    Parameters
    ----------
    h5_file : Path
        Path to h5 file with statistics
    thresholds : Thresholds
        Thresholds object with thresholds for qc
    output_path : str
        Path to output directory

    Returns
    -------
    tuple[dict, list[Path]]
        Parameters for report, list of temporary files to be removed after report generation
    """
    # list of files to be removed after report generation
    tmp_files = []

    # prepare parameters for report: add statistics
    parameters = dict(statistics_h5=h5_file)

    # add thresholds to parameters
    for threshold_name, threshold_value in vars(thresholds).items():
        parameters[threshold_name + "_threshold"] = threshold_value

    # add plots to parameters
    cbc_umi_png = cbc_umi_plot(h5_file, output_path)
    parameters["cbc_umi_png"] = cbc_umi_png
    tmp_files.append(cbc_umi_png)

    insert_length_png = plot_insert_length_histogram(h5_file, output_path)
    parameters["insert_length_png"] = insert_length_png
    tmp_files.append(insert_length_png)

    mean_insert_quality_histogram_png = plot_mean_insert_quality_histogram(
        h5_file, output_path
    )
    parameters["mean_insert_quality_histogram_png"] = mean_insert_quality_histogram_png
    tmp_files.append(mean_insert_quality_histogram_png)

    quality_per_position_png = plot_quality_per_position(h5_file, output_path)
    parameters["quality_per_position_png"] = quality_per_position_png
    tmp_files.append(quality_per_position_png)

    return parameters, tmp_files


def generate_report(
    parameters, output_path, tmp_files: list[Path], sample_name: str
) -> Path:
    """
    Generate report based on jupyter notebook template.

    Parameters
    ----------
    parameters : dict
        Parameters for report
    output_path : str
        Path to output directory
    tmp_files : list[Path]
        List of temporary files to be removed after report generation
    sample_name : str
        Sample name to be included as a prefix in the output files

    Returns
    -------
    Path
        Path to generated report
    """
    # define outputs
    output_report_html = Path(output_path) / (
        sample_name + OutputFiles.HTML_REPORT.value
    )
    output_report_ipynb = Path(output_path) / (sample_name + OutputFiles.NOTEBOOK.value)
    tmp_files.append(output_report_ipynb)

    # inject parameters and run notebook
    parameters = {
        k: str(v) if isinstance(v, Path) else v for k, v in parameters.items()
    }
    papermill.execute_notebook(
        input_path=str(TEMPLATE_NOTEBOOK),
        output_path=str(output_report_ipynb),
        parameters=parameters,
        kernel_name="python3",
    )

    # convert to html
    notebook = nbformat.read(str(output_report_ipynb), as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True
    (body, resources) = html_exporter.from_notebook_node(notebook)

    with open(output_report_html, "w") as f:
        f.write(body)

    # edit html for readability
    modify_jupyter_notebook_html(output_report_html)

    # remove temporary files - png and ipynb files
    for temp_file in tmp_files:
        if temp_file.is_file():
            temp_file.unlink()

    return output_report_html


def main():
    # parse args from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--sample-name",
        type=str,
        required=True,
        help="Sample name to be included in the output files",
    )
    parser.add_argument(
        "--trimmer-stats",
        type=str,
        required=True,
        help="Path to Trimmer stats csv file",
    )
    parser.add_argument(
        "--trimmer-histogram",
        type=str,
        required=True,
        nargs="+",
        help="Path to Trimmer histogram csv files. Multiple files are supported, pass them with space separated.",
    )
    parser.add_argument(
        "--trimmer-failure-codes",
        type=str,
        required=True,
        help="Path to Trimmer failure codes csv file",
    )
    parser.add_argument(
        "--sorter-stats",
        type=str,
        required=True,
        help="Path to Sorter stats csv file",
    )
    parser.add_argument(
        "--star-stats", type=str, required=True, help="Path to STAR stats file"
    )
    parser.add_argument(
        "--insert",
        type=str,
        required=True,
        help="Path to insert .fastq.gz file",
    )
    parser.add_argument(
        "--star-reads-per-gene",
        type=str,
        required=True,
        help="Path to STAR ReadsPerGene.out.tab file",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--pass-trim-rate", type=float, required=True, help="Minimal %trimmed"
    )
    parser.add_argument(
        "--read-length", type=int, required=True, help="Expected read length"
    )
    parser.add_argument(
        "--fraction-below-read-length",
        type=float,
        required=True,
        help="Fraction of reads below read length threshold",
    )
    parser.add_argument(
        "--percent-aligned",
        type=float,
        required=True,
        help="Minimal % of reads aligned",
    )

    args = parser.parse_args()

    # create Inputs and Thresholds objects
    inputs = Inputs(
        args.trimmer_stats,
        args.trimmer_histogram,
        args.trimmer_failure_codes,
        args.sorter_stats,
        args.star_stats,
        args.star_reads_per_gene,
        args.insert,
    )
    thresholds = Thresholds(
        args.pass_trim_rate,
        args.read_length,
        args.fraction_below_read_length,
        args.percent_aligned,
    )
    # run single_cell_qc
    single_cell_qc(
        input_files=inputs,
        output_path=args.output_path,
        thresholds=thresholds,
        sample_name=args.sample_name,
    )


if __name__ == "__main__":
    main()
