import os
from pathlib import Path
import subprocess
from argparse import ArgumentParser
from typing import List, Union


from ugvc.pipelines.single_cell_qc.collect_statistics import collect_statistics, extract_statistics_table
from ugvc.pipelines.single_cell_qc.create_plots import (
    cbc_umi_plot,
    plot_mean_r2_quality_histogram,
    plot_quality_per_position,
    plot_r2_length_histogram,
)
from ugvc.pipelines.single_cell_qc.sc_qc_dataclasses import Inputs, Thresholds
from ugvc.utils.misc_utils import modify_jupyter_notebook_html


def single_cell_qc(input_files: Inputs, output_path: str, thresholds: Thresholds):

    h5_file = collect_statistics(input_files, output_path)
    extract_statistics_table(h5_file)

    params, tmp_files = prepare_parameters_for_report(h5_file, thresholds, output_path)
    output_report_html = generate_report(params, output_path, tmp_files=tmp_files)
    # TODO: export h5_file and report html to papyrus


def prepare_parameters_for_report(h5_file: str, thresholds: Thresholds, output_path: str) -> Union[dict, List[str]]:
    # list of files to be removed after report generation
    tmp_files = []

    # prepare parameters for report: add statistics
    parameters = dict(statistics_h5=h5_file)

    # add thresholds to parameters
    for threshold_name, threshold_value in vars(thresholds).items():
        parameters[threshold_name+"_threshold"] = threshold_value

    # add plots to parameters
    cbc_umi_png = cbc_umi_plot(h5_file, output_path)
    parameters["cbc_umi_png"] = cbc_umi_png
    tmp_files.append(cbc_umi_png)

    r2_length_png = plot_r2_length_histogram(h5_file, output_path)
    parameters["r2_length_png"] = r2_length_png
    tmp_files.append(r2_length_png)

    mean_r2_quality_histogram_png = plot_mean_r2_quality_histogram(h5_file, output_path)
    parameters["mean_r2_quality_histogram_png"] = mean_r2_quality_histogram_png
    tmp_files.append(mean_r2_quality_histogram_png)

    quality_per_position_png = plot_quality_per_position(h5_file, output_path)
    parameters["quality_per_position_png"] = quality_per_position_png
    tmp_files.append(quality_per_position_png)

    return parameters, tmp_files


def generate_report(parameters, output_path, tmp_files=[]) -> str:
    # define outputs
    output_report_html = os.path.join(output_path, f"single_cell_qc_report.html")
    output_report_ipynb = os.path.join(output_path, f"single_cell_qc_report.ipynb")
    tmp_files.append(output_report_ipynb)
    template_notebook = Path.cwd() / 'ugvc' / 'reports' / 'single_cell_qc_report.ipynb'

    # inject parameters and run notebook
    papermill_params = f"{' '.join([f'-p {k} {v}' for k, v in parameters.items()])}"
    papermill_cmd = f"papermill {template_notebook} {output_report_ipynb} {papermill_params} -k python3"
    subprocess.check_call(papermill_cmd.split())

    # convert to html
    subprocess.check_call(
        f"jupyter nbconvert {output_report_ipynb} --to html --no-input".split()
    )

    # edit html for readability
    modify_jupyter_notebook_html(output_report_html)

    # remove temporary files - png and ipynb files
    for temp_file in tmp_files:
        if os.path.isfile(temp_file):
            os.remove(temp_file)

    return output_report_html


if __name__ == "__main__":
    # parse args from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--trimmer-stats",
        type=str,
        required=True,
        help="path to Trimmer stats csv file",
    )
    parser.add_argument(
        "--trimmer-histogram",
        type=str,
        required=True,
        nargs="+",
        help="path to Trimmer histogram csv files. Multiple files are supported, pass them with space separated.",
    )
    parser.add_argument(
        "--trimmer-failure-codes",
        type=str,
        required=True,
        help="path to Trimmer failure codes csv file",
    )
    parser.add_argument(
        "--sorter-stats",
        type=str,
        required=True,
        help="path to Sorter stats csv file",
    )
    parser.add_argument(
        "--star-stats", type=str, required=True, help="path to STAR stats file"
    )
    parser.add_argument(
        "--r2-subsample",
        type=str,
        required=True,
        help="path to R2 subsample .fastq.gz file",
    )
    parser.add_argument(
        "--star-reads-per-gene",
        type=str,
        required=True,
        help="path to STAR ReadsPerGene.out.tab file",
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="path to output directory"
    )
    parser.add_argument(
        "--pass-trim-rate", type=float, required=True, help="minimal %trimmed"
    )
    parser.add_argument(
        "--read-length", type=int, required=True, help="expected read length"
    )
    parser.add_argument(
        "--fraction-below-read-length",
        type=float,
        required=True,
        help="fraction of reads below read length",
    )
    parser.add_argument(
        "--percent-aligned",
        type=float,
        required=True,
        help="minimal % of reads aligned",
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
        args.r2_subsample,
    )
    thresholds = Thresholds(
        args.pass_trim_rate,
        args.read_length,
        args.fraction_below_read_length,
        args.percent_aligned,
    )
    # run single_cell_qc
    single_cell_qc(
        input_files=inputs, output_path=args.output_path, thresholds=thresholds
    )
