import os
from os.path import join as pjoin
from test import get_resource_dir

import joblib
import pandas as pd

from ugvc import logger
from ugvc.mrd.srsnv_plotting_utils import SRSNVReport  # , default_LoD_filters, retention_noise_and_mrd_lod_simulation
from ugvc.mrd.srsnv_training_utils import SRSNVTrain

inputs_dir = get_resource_dir(__file__)


def test_create_report(tmpdir):
    """test the create_report_plots function"""

    # TODO: add train with a small model (low tree num)

    # Read files
    base_name = "402572-CL10377.sample30K."

    model_joblib_file = pjoin(
        inputs_dir,
        f"{base_name}model.joblib",
    )
    df_file = pjoin(
        inputs_dir,
        f"{base_name}featuremap_df.parquet",
    )
    df_mrd_simulation_file = pjoin(
        inputs_dir,
        f"{base_name}test.df_mrd_simulation.parquet",
    )

    model_joblib = joblib.load(model_joblib_file)
    df_mrd_simulation = pd.read_parquet(df_mrd_simulation_file)
    data_df = pd.read_parquet(df_file)

    statistics_h5_file = pjoin(
        tmpdir,
        f"{base_name}test.statistics.h5",
    )
    statistics_json_file = pjoin(
        tmpdir,
        f"{base_name}test.statistics.json",
    )

    # report_name = "test"

    # Add needed columns to the dataframe
    class SRSNVTRainMock(SRSNVTrain):
        def __init__(
            self,
            featuremap_df: pd.DataFrame,
            classifiers: list,
            params: dict,
        ):
            k_folds = params["num_CV_folds"]
            # Check whether using cross validation:
            assert k_folds > 0, f"k_folds should be > 0, got {k_folds=}"
            if k_folds == 1:
                self.use_CV = False
            else:
                self.use_CV = True
                logger.info(f"Using cross validation with {k_folds} folds")
            self.k_folds = k_folds
            self.chroms_to_folds = params["chroms_to_folds"]

            self.featuremap_df = featuremap_df
            self.classifiers = classifiers
            self.numerical_features = params["numerical_features"]
            self.categorical_features_dict = params["categorical_features_dict"]
            self.categorical_features_names = params["categorical_features_names"]
            self.columns = self.numerical_features + self.categorical_features_names

            self.ppmSeq_adapter_version = params["adapter_version"]
            self.start_tag_col = params["start_tag_col"]
            self.end_tag_col = params["end_tag_col"]
            # self.pipeline_version = params['pipeline_version']
            # self.docker_image = params['docker_image']

    srsnvtrain = SRSNVTRainMock(
        featuremap_df=data_df, classifiers=model_joblib["models"], params=model_joblib["params"]
    )
    srsnvtrain.add_is_mixed_to_featuremap_df()
    srsnvtrain.add_predictions_to_featuremap_df()
    data_df = srsnvtrain.featuremap_df
    # Add qual column to data_df
    data_df["qual"] = (
        data_df["ML_qual_1_test"]
        .apply(model_joblib["quality_interpolation_function"])
        .mask(data_df["ML_qual_1_test"].isna())
    )  # nan values in ML_qual_1_test (for reads not in test set) will be nan in qual

    # lod params
    lod_label = "LoD @ 99% specificity,     90% sensitivity (estimated)    \nsignature size 10000,     30x coverage"
    c_lod = "LoD_90"

    # Create report

    base_name_for_report = base_name if base_name.endswith(".") else f"{base_name}."
    SRSNVReport(
        models=model_joblib["models"],
        data_df=data_df,
        params=model_joblib["params"],
        out_path=tmpdir,
        base_name=base_name_for_report,
        lod_filters=model_joblib["params"]["lod_filters"],  # , self.lod_filters,
        lod_label=lod_label,  # self.lod_label,
        c_lod=c_lod,  # self.c_lod,
        df_mrd_simulation=df_mrd_simulation,  # self.df_mrd_simulation,
        ML_qual_to_qual_fn=model_joblib["quality_interpolation_function"],  # self.quality_interpolation_function,
        statistics_h5_file=statistics_h5_file,
        statistics_json_file=statistics_json_file,
        rng=None,  # self.rng,
        raise_exceptions=True,
    ).create_report()

    # check if all plots are created
    report_plot_names = [
        "LoD_curve",
        "SHAP_beeswarm",
        "SHAP_importance",
        "calibration_fn_with_hist",
        "logit_histogram",
        "qual_histogram",
        "qual_vs_ppmSeq_tags_table",
        "training_progress",
        "trinuc_stats",
    ]

    for col in model_joblib["params"]["numerical_features"]:
        report_plot_names.append(f"qual_per_{col}")

    report_plot_filenames = [f"{base_name}{fname}.png" for fname in report_plot_names]

    missing_plots = []
    for pname in report_plot_filenames:
        if not os.path.isfile(pjoin(tmpdir, pname)):
            missing_plots.append(pname)
    assert not missing_plots, f"Missing plots: {missing_plots}"

    assert os.path.isfile(pjoin(tmpdir, f"{base_name}single_read_snv.applicationQC.h5")), "Missing annotationQC.h5 file"
    #  TODO: check that keys in annotationQC.h5 are correct

    for pname in [statistics_h5_file, statistics_json_file]:
        assert os.path.isfile(pname), f"Missing statistics file: {pname}"
