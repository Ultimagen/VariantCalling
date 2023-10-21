import json
import os
from os.path import basename
from os.path import join as pjoin
from test import get_resource_dir

import joblib

from ugvc.mrd.srsnv_plotting_utils import create_report_plots

inputs_dir = get_resource_dir(__file__)


def test_create_report_plots(tmpdir):
    """test the create_report_plots function"""

    # TODO: add train with a small model (low tree num)

    base_name = "balanced_ePCR_LA5_LA6_333_LuNgs_08."

    model_file = pjoin(
        inputs_dir,
        f"{base_name}model.joblib",
    )
    y_file = pjoin(
        inputs_dir,
        f"{base_name}y_test.parquet",
    )
    X_file = pjoin(
        inputs_dir,
        f"{base_name}X_test.parquet",
    )
    params_file = pjoin(
        inputs_dir,
        f"{base_name}params.json",
    )

    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    params["fp_regions_bed_file"] = pjoin(
        inputs_dir,
        "ug_hcr_18.bed",
    )
    params["sorter_json_stats_file"] = pjoin(
        inputs_dir,
        f"{base_name}json",
    )
    params["adapter_version"] = "LA_v5and6"

    local_params_file = pjoin(tmpdir, basename(params_file).replace(".json", ".local.json"))
    with open(local_params_file, "w", encoding="utf-8") as f:
        json.dump(params, f)

    statistics_h5_file = pjoin(
        tmpdir,
        f"{base_name}test.statistics.h5",
    )

    statistics_json_file = pjoin(
        tmpdir,
        f"{base_name}test.statistics.json",
    )

    report_name = "test"
    create_report_plots(
        model_file,
        X_file,
        y_file,
        local_params_file,
        report_name=report_name,
        out_path=tmpdir,
        base_name=base_name,
        lod_filters=None,
        mrd_simulation_dataframe_file=None,
        statistics_h5_file=statistics_h5_file,
        statistics_json_file=statistics_json_file,
    )

    # check if all plots are created
    report_plots = [
        f"{base_name}{report_name}.LoD_curve.png",
        f"{base_name}{report_name}.ROC_curve.png",
        f"{base_name}{report_name}.confusion_matrix.png",
        f"{base_name}{report_name}.precision_recall_qual.png",
        f"{base_name}{report_name}.qual_density.png",
        f"{base_name}{report_name}.observed_qual.png",
        f"{base_name}{report_name}.ML_qual_hist.png",
    ]

    model = joblib.load(model_file)
    for f in model.feature_names_in_:
        report_plots.append(f"{base_name}{report_name}.qual_per_{f}.png")

    if (
        "strand_ratio_category_end" in model.feature_names_in_
        and "strand_ratio_category_start" in model.feature_names_in_
    ):
        report_plots.append(f"{base_name}{report_name}.balanced_strand_mixed_cs.png")
        report_plots.append(f"{base_name}{report_name}.balanced_strand_mixed_non_cs.png")
        report_plots.append(f"{base_name}{report_name}.balanced_strand_mixed_cs.png")
        report_plots.append(f"{base_name}{report_name}.balanced_strand_non_mixed_non_cs.png")
        report_plots.append(f"{base_name}{report_name}.balanced_strand_fpr.png")
        report_plots.append(f"{base_name}{report_name}.balanced_strand_recalls.png")

    for pname in report_plots:
        assert os.path.isfile(pjoin(tmpdir, f"{pname}")), f"Missing report plot: {pname}"

    for pname in [statistics_h5_file, statistics_json_file]:
        assert os.path.isfile(pname), f"Missing statistics file: {pname}"
