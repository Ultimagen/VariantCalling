from __future__ import annotations

import json
import os
from os.path import join as pjoin

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import colors
from scipy.interpolate import interp1d
from scipy.stats import binom
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tqdm import tqdm

from ugvc import logger
from ugvc.mrd.balanced_strand_utils import BalancedStrandAdapterVersions
from ugvc.mrd.featuremap_utils import FeatureMapFields
from ugvc.utils.exec_utils import print_and_execute
from ugvc.utils.metrics_utils import convert_h5_to_json, read_effective_coverage_from_sorter_json
from ugvc.utils.misc_utils import filter_valid_queries
from ugvc.utils.plotting_utils import set_default_plt_rc_params
from ugvc.vcfbed.filter_bed import count_bases_in_bed_file

edist_filter = f"{FeatureMapFields.X_EDIST.value} <= 5"
HQ_SNV_filter = f"{FeatureMapFields.X_SCORE.value} >= 7.9"
CSKP_SNV_filter = f"{FeatureMapFields.X_SCORE.value} >= 10"
read_end_filter = (
    f"{FeatureMapFields.X_INDEX.value} > 12 and "
    f"{FeatureMapFields.X_INDEX.value} < ({FeatureMapFields.X_LENGTH.value} - 12)"
)
mixed_read_filter = "is_mixed"  # TODO use adapter_version
default_LoD_filters = {
    "no_filter": f"{FeatureMapFields.X_SCORE.value} >= 0",
    "HQ_SNV": f"{HQ_SNV_filter} and {edist_filter}",
    "CSKP": f"{CSKP_SNV_filter} and {edist_filter}",
    "HQ_SNV_trim_ends": f"{HQ_SNV_filter} and {edist_filter} and {read_end_filter}",
    "CSKP_trim_ends": f"{CSKP_SNV_filter} and {edist_filter} and {read_end_filter}",
    "HQ_SNV_mixed_only": f"{HQ_SNV_filter} and {edist_filter} and {mixed_read_filter}",
    "CSKP_mixed_only": f"{CSKP_SNV_filter} and {edist_filter} and {mixed_read_filter}",
    "HQ_SNV_trim_ends_mixed_only": f"{HQ_SNV_filter} and {edist_filter} and {read_end_filter} and {mixed_read_filter}",
    "CSKP_trim_ends_mixed_only": f"{CSKP_SNV_filter} and {edist_filter} and {read_end_filter} and {mixed_read_filter}",
}
TP_READ_RETENTION_RATIO = "tp_read_retention_ratio"
FP_READ_RETENTION_RATIO = "fp_read_retention_ratio"
RESIDUAL_SNV_RATE = "residual_snv_rate"


def create_data_for_report(
    classifier: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
):
    """create the data needed for the report plots

    Parameters
    ----------
    classifier : xgb.XGBClassifier
        the trained ML model
    X : pd.DataFrame
        input data with features for the classifier
    y : pd.DataFrame
        labels

    Returns
    -------
    df : pd.DataFrame
        dataset with input features, model predicions and probabilities
    df_tp : pd.DataFrame
        TP subset of df
    df_fp : pd.DataFrame
        FP subset of df
    max_score : float
        maximal ML model score
    cls_features : list
        list of input features for the ML model
    fprs : dict
        list of false positive rates per ML score per label
    recalls : dict
        list of false positive rates per ML score per label
    """

    cls_features = classifier.feature_names_in_
    probs = classifier.predict_proba(X[cls_features])
    predictions = classifier.predict(X[cls_features])
    quals = -10 * np.log10(1 - probs)
    predictions_df = pd.DataFrame(y)
    labels = np.unique(y["label"].astype(int))
    for label in labels:
        predictions_df[f"ML_prob_{label}"] = probs[:, label]
        predictions_df[f"ML_qual_{label}"] = quals[:, label]
        predictions_df[f"ML_prediction_{label}"] = predictions[:, label]
    # TODO: write the code below with .assign rather than concat
    df = pd.concat([X, predictions_df], axis=1)
    # TODO: use the information from adapter_version instead of this patch
    if "strand_ratio_category_end" in df and "strand_ratio_category_start" in df:
        df = df.assign(
            is_mixed=((df["strand_ratio_category_end"] == "MIXED") & (df["strand_ratio_category_start"] == "MIXED"))
        )

    df_tp = df.query("label == True")
    df_fp = df.query("label == False")

    fprs = {}
    recalls = {}
    max_score = -1
    for label in labels:
        fprs[label] = []
        recalls[label] = []
        score = f"ML_qual_{label}"
        max_score = np.max((int(np.ceil(df[score].max())), max_score))
        gtr = df["label"] == label
        fprs_, recalls_ = precision_recall_curve(df[score], max_score=max_score, y_true=gtr)
        fprs[label].append(fprs_)
        recalls[label].append(recalls_)

    return df, df_tp, df_fp, max_score, cls_features, fprs, recalls


def srsnv_report(
    out_path,
    out_basename,
    report_name,
    model_file,
    params_file,
    simple_pipeline=None,
):
    if len(out_basename) > 0 and not out_basename.endswith("."):
        out_basename += "."
    reportfile = pjoin(out_path, f"{out_basename}{report_name}_report.ipynb")
    reporthtml = pjoin(out_path, f"{out_basename}{report_name}_report.html")

    [
        output_roc_plot,
        output_LoD_plot,
        output_cm_plot,
        output_precision_recall_qual,
        output_qual_density,
        output_obsereved_qual_plot,
        output_ML_qual_hist,
        output_qual_per_feature,
        output_balanced_strand_hists,
        output_balanced_strand_fpr,
        output_balanced_strand_recalls,
    ] = _get_plot_paths(report_name, out_path=out_path, out_basename=out_basename)

    template_notebook = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "reports/srsnv_report.ipynb",
    )
    papermill_command = f"papermill {template_notebook} {reportfile} \
-p report_name {report_name} \
-p model_file {model_file} \
-p params_file {params_file} \
-p output_roc_plot {output_roc_plot} \
-p output_LoD_plot {output_LoD_plot} \
-p output_cm_plot {output_cm_plot} \
-p output_precision_recall_qual {output_precision_recall_qual} \
-p output_qual_density {output_qual_density} \
-p output_obsereved_qual_plot {output_obsereved_qual_plot} \
-p output_ML_qual_hist {output_ML_qual_hist} \
-p output_qual_per_feature {output_qual_per_feature} \
-p output_bepcr_hists {output_balanced_strand_hists} \
-p output_bepcr_fpr {output_balanced_strand_fpr} \
-p output_bepcr_recalls {output_balanced_strand_recalls} \
-k python3"
    jupyter_nbconvert_command = f"jupyter nbconvert {reportfile} --output {reporthtml} --to html --no-input"

    print_and_execute(papermill_command, simple_pipeline=simple_pipeline, module_name=__name__)
    print_and_execute(jupyter_nbconvert_command, simple_pipeline=simple_pipeline, module_name=__name__)


def plot_ROC_curve(
    df: pd.DataFrame,
    df_tp: pd.DataFrame,
    df_fp: pd.DataFrame,
    ML_score: str,
    adapter_version: str,
    title: str = "",
    output_filename: str = None,
):
    """generate and save ROC curve plot

    Parameters
    ----------
    df : pd.DataFrame
        data set with features, labels and quals of the model
    df_tp : pd.DataFrame
        subset of df: rows with label TP
    df_fp : pd.DataFrame
        subset of df: reads with label FP
    ML_score : str
        ML model score name (column in df),
    adapter_version : str
        adapter version, indicates if input featuremap is from balanced ePCR data
    title : str, optional
        title for the ROC curve plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    """
    set_default_plt_rc_params()

    label_dict = {
        f"{FeatureMapFields.X_SCORE.value}": "FeatureMap",
        ML_score: "ML model",
    }

    xvars = [f"{FeatureMapFields.X_SCORE.value}", ML_score]
    querystrs = [
        f"{FeatureMapFields.X_SCORE.value}>-1",
        f"{FeatureMapFields.X_SCORE.value}>-1",
    ]
    namestrs = [f"{FeatureMapFields.X_SCORE.value}", ML_score]

    recall = {}
    precision = {}

    if adapter_version in [av.value for av in BalancedStrandAdapterVersions]:
        xvars.append(f"{FeatureMapFields.X_SCORE.value}")
        querystrs.append("is_mixed == True & X_SCORE>-1")
        namestrs.append(f"{FeatureMapFields.X_SCORE.value}_mixed")
        label_dict[f"{FeatureMapFields.X_SCORE.value}_mixed"] = "FeatureMap"

    for xvar, querystr, namestr in zip(xvars, querystrs, namestrs):
        thresholds = np.linspace(0, df[xvar].quantile(0.99), 100)
        recall[namestr] = {}
        precision[namestr] = {}
        for thresh in thresholds:
            recall[namestr][thresh] = (df_tp.query(querystr)[xvar] > thresh).sum() / df_tp.shape[0]
            total_fp = (df_fp.query(querystr)[xvar] >= thresh).sum()
            total_tp = (df_tp.query(querystr)[xvar] >= thresh).sum()
            precision[namestr][thresh] = total_tp / (total_tp + total_fp)

    plt.figure(figsize=(12, 8))
    for item in recall.items():
        label = item[0]
        auc = -np.trapz(list(precision[label].values()), list(recall[label].values()))
        plt.plot(
            recall[label].values(),
            precision[label].values(),
            "-o",
            label=f"{label_dict[label]} ({label}) - AUC={auc:.2f}",
        )
        plt.xlabel("Recall (TP over TP+FN)", fontsize=24)
        plt.ylabel("Precision (total TP over FP+TP)", fontsize=24)
        legend_handle = plt.legend(fontsize=24, fancybox=True, framealpha=0.95)
        title_handle = plt.title(title, fontsize=24)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


# pylint: disable=too-many-arguments
def retention_noise_and_mrd_lod_simulation(
    df: pd.DataFrame,
    single_sub_regions: str,
    sorter_json_stats_file: str,
    training_set_downsampling_rate: float,
    lod_filters: dict,
    sensitivity_at_lod: float = 0.90,
    specificity_at_lod: float = 0.99,
    simulated_signature_size: int = 10_000,
    simulated_coverage: int = 30,
    minimum_number_of_read_for_detection: int = 2,
    output_dataframe_file: str = None,
):
    """Estimate the MRD LoD based on the FeatureMap dataframe and the LoD simulation params

    Parameters
    ----------
    df : pd.DataFrame
        dataset with features, labels, and quals of the model.
    single_sub_regions : str
        a path to the single substitutions (FP) regions bed file.
    sorter_json_stats_file : str
        a path to the cram statistics file.
    training_set_downsampling_rate : float
        the size of the dataset / # of reads in single sub featuremap after intersection with single_sub_regions.
    lod_filters : dict
        filters for LoD simulation.
    sensitivity_at_lod: float
        The sensitivity at which LoD is estimated.
    specificity_at_lod: float
        The specificity at which LoD is estimated.
    simulated_signature_size: int
        The size of the simulated signature.
    simulated_coverage: int
        The simulated coverage.
    minimum_number_of_read_for_detection: int
        The minimum number of reads required for detection in the LoD simulation. Default 2.
    output_dataframe_file: str, optional
        Path to output dataframe file. If None, don't save.

    Returns
    -------
    df_mrd_sim: pd.DataFrame
        The estimated LoD parameters per filter.
    lod_label: str
        Label for the legend in the Lod plot.
    c_lod: str
        lod column name
    """

    # Filter queries in LoD dict that raise an error (e.g. contain non existing fields like "is_mixed")
    lod_filters = filter_valid_queries(df_test=df.head(10), queries=lod_filters)

    # apply filters
    df = df.assign(
        **{
            filter_name: df.eval(filter_query)
            for filter_name, filter_query in tqdm(lod_filters.items())
            if filter_query is not None
        }
    )
    df_fp = df.query("label == 0")
    df_tp = df.query("label == 1")

    # count the # of bases in region
    n_bases_in_region = count_bases_in_bed_file(single_sub_regions)

    # get coverage statistics
    (
        mean_coverage,
        ratio_of_reads_over_mapq,
        ratio_of_bases_in_coverage_range,
        _,
        _,
    ) = read_effective_coverage_from_sorter_json(sorter_json_stats_file)

    # calculate the read filter correction factor (TP reads pre-filtered from the FeatureMap)
    read_filter_correction_factor = 1
    if (f"{FeatureMapFields.FILTERED_COUNT.value}" in df_tp) and (f"{FeatureMapFields.READ_COUNT.value}" in df_tp):
        read_filter_correction_factor = (df_tp[f"{FeatureMapFields.FILTERED_COUNT.value}"]).sum() / df_tp[
            f"{FeatureMapFields.READ_COUNT.value}"
        ].sum()
    else:
        logger.warning(
            f"{FeatureMapFields.FILTERED_COUNT.value} or {FeatureMapFields.READ_COUNT.value} no in dataset, \
                read_filter_correction_factor = 1 in LoD"
        )

    # calculate the error rate (residual SNV rate) normalization factors
    n_noise_reads = df_fp.shape[0]
    n_signal_reads = df_tp.shape[0]
    effective_bases_covered = (
        mean_coverage
        * n_bases_in_region
        * training_set_downsampling_rate
        * ratio_of_reads_over_mapq
        * ratio_of_bases_in_coverage_range
        * read_filter_correction_factor
    )
    residual_snv_rate_no_filter = n_noise_reads / effective_bases_covered
    ratio_filtered_prior_to_featuremap = ratio_of_reads_over_mapq * read_filter_correction_factor
    logger.info(
        f"n_noise_reads {n_noise_reads}, n_signal_reads {n_signal_reads}"
        f", effective_bases_covered {effective_bases_covered}, "
        f"residual_snv_rate_no_filter {residual_snv_rate_no_filter}"
    )

    # Calculate simulated LoD definitions and correction factors
    effective_signature_bases_covered = int(
        simulated_coverage * simulated_signature_size
    )  # The ratio_filtered_prior_to_featuremap is not taken into account here, but later in the binomial calculation

    # create a dataframe with the LoD parameters per filter
    df_mrd_simulation = pd.concat(
        (
            (df_fp[list(lod_filters.keys())].sum() / n_noise_reads).rename(FP_READ_RETENTION_RATIO),
            (df_tp[list(lod_filters.keys())].sum() / n_signal_reads * ratio_filtered_prior_to_featuremap).rename(
                TP_READ_RETENTION_RATIO
            ),
        ),
        axis=1,
    )

    # remove filters with a low TP read retention ratio
    MIN_TP_RETENTION = 0.01
    df_mrd_simulation = df_mrd_simulation[df_mrd_simulation[TP_READ_RETENTION_RATIO] > MIN_TP_RETENTION]

    # Assign the residual SNV rate per filter
    df_mrd_simulation.loc[:, RESIDUAL_SNV_RATE] = (
        df_mrd_simulation[FP_READ_RETENTION_RATIO]
        * residual_snv_rate_no_filter
        / df_mrd_simulation[TP_READ_RETENTION_RATIO]
    )

    # Calculate the minimum number of reads required for detection, per filter, assuming a binomial distribution
    # The probability of success is the product of the read retention ratio and the residual SNV rate
    df_mrd_simulation = df_mrd_simulation.assign(
        min_reads_for_detection=np.ceil(
            binom.ppf(
                n=int(effective_signature_bases_covered),
                p=df_mrd_simulation[TP_READ_RETENTION_RATIO] * df_mrd_simulation[RESIDUAL_SNV_RATE],
                q=specificity_at_lod,
            )
        )
        .clip(min=minimum_number_of_read_for_detection)
        .astype(int),
    )

    # Simulate the LoD per filter, assuming a binomial distribution
    # The simulation is done by drawing the expected number of reads in the Qth percentile, where the percentile is
    # (1-sensitivity), for a range of tumor frequencies (tf_sim). The lowest tumor fraction that passes the prescribed
    # minimum number of reads for detection is the LoD.
    tf_sim = np.logspace(-8, 0, 500)
    c_lod = f"LoD_{sensitivity_at_lod*100:.0f}"
    df_mrd_simulation = df_mrd_simulation.join(
        df_mrd_simulation.apply(
            lambda row: tf_sim[
                np.argmax(
                    binom.ppf(
                        q=1 - sensitivity_at_lod,
                        n=int(effective_signature_bases_covered),
                        p=row[TP_READ_RETENTION_RATIO] * (tf_sim + row[RESIDUAL_SNV_RATE]),
                    )
                    >= row["min_reads_for_detection"]
                )
            ],
            axis=1,
        ).rename(c_lod)
    )

    lod_label = f"LoD @ {specificity_at_lod*100:.0f}% specificity, \
    {sensitivity_at_lod*100:.0f}% sensitivity (estimated)\
    \nsignature size {simulated_signature_size}, \
    {simulated_coverage}x coverage"

    if output_dataframe_file:
        df_mrd_simulation.to_parquet(output_dataframe_file)

    return df_mrd_simulation, lod_filters, lod_label, c_lod


def plot_LoD(
    df_mrd_sim: pd.DataFrame,
    lod_label: str,
    c_lod: str,
    filters: dict,
    adapter_version: str,
    min_LoD_filter: str,
    title: str = "",
    output_filename: str = None,
):
    """generates and saves the LoD plot

    Parameters
    ----------
    df_mrd_sim : pd.DataFrame
        the estimated LoD parameters per filter
    lod_label : str
        label for the LoD axis in the Lod plot
    c_lod : str
        lod column name
    filters : dict
        filters applied on data to estimate LoD
    adapter_version : str
        adapter version, indicates if input featuremap is from balanced ePCR data
    min_LoD_filter : str
        the filter which minimizes the LoD
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None

    """
    set_default_plt_rc_params()

    # TODO: a less patchy solution, perhaps accept ml_filters as a separate input
    ML_filters = {i: filters[i] for i in filters if i[:2] == "ML" and i in df_mrd_sim.index}

    fig = plt.figure(figsize=(20, 12))

    filters_list = [
        list(ML_filters),
        ["no_filter"],
        ["HQ_SNV"],
    ]
    markers_list = ["o", "*", "D"]
    labels_list = [
        "ML model",
        "No filter",
        "HQ_SNV and Edit Dist <= 5",
    ]
    edgecolors_list = ["r", "r", "r"]
    msize_list = [150, 150, 150]
    if adapter_version in [av.value for av in BalancedStrandAdapterVersions]:
        filters_list.append(["HQ_SNV_mixed_only"])
        markers_list.append(">")
        labels_list.append("HQ_SNV and Edit Dist <= 5 (mixed only)")
        edgecolors_list.append("r")
        msize_list.append(150)

        filters_list.append(["CSKP_mixed_only"])
        markers_list.append("s")
        labels_list.append("CSKP and Edit Dist <= 5 (mixed only)")
        edgecolors_list.append("r")
        msize_list.append(150)

    best_lod = df_mrd_sim.loc[
        [item for sublist in filters_list for item in sublist if item in df_mrd_sim.index.values], c_lod
    ].min()  # best LoD across all plotted results

    for f, marker, label, edgecolor, markersize in zip(
        filters_list, markers_list, labels_list, edgecolors_list, msize_list
    ):
        if f not in df_mrd_sim.index:
            continue
        df_tmp = df_mrd_sim.loc[f]
        plt.plot(
            df_tmp[TP_READ_RETENTION_RATIO],
            df_tmp[RESIDUAL_SNV_RATE],
            c="k",
            alpha=0.3,
        )
        best_lod_filter = df_tmp[c_lod].min()
        plt.scatter(
            df_tmp[TP_READ_RETENTION_RATIO],
            df_tmp[RESIDUAL_SNV_RATE],
            c=df_tmp[c_lod],
            marker=marker,
            edgecolor=edgecolor,
            label=f"{label}, best LoD: {best_lod_filter:.1E}".replace("E-0", "E-"),
            s=markersize,
            zorder=markersize,
            norm=colors.LogNorm(
                vmin=best_lod,
                vmax=best_lod * 10,
            ),
        )
    plt.xlabel("Base retention ratio on HOM SNVs")
    plt.ylabel("Residual SNV rate")
    plt.yscale("log")
    title_handle = plt.title(title)
    legend_handle = plt.legend(fancybox=True, framealpha=0.95)

    cbar = plt.colorbar()
    cbar.set_label(label=lod_label, fontsize=20)
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # cbar.ax.yaxis.set_major_formatter(formatter)

    fig.text(
        0.5,
        0.01,
        f"ML qual threshold for min LoD: {min_LoD_filter}",
        ha="center",
        bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
    )
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_confusion_matrix(
    df: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    prediction_column_name="ML_prediction_1",
):
    """generates and saves confusion matrix

    Parameters
    ----------
    df : pd.DataFrame
        data set with labels and model predictions
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    prediction_column_name : str, optional
        column name of the model predictions, by default "ML_prediction_1"
    """
    set_default_plt_rc_params()

    cm = confusion_matrix(y_true=df["label"], y_pred=df[prediction_column_name])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize by rows - true labels
    plt.figure(figsize=(4, 3))
    plt.grid(False)
    ax = sns.heatmap(
        cm_norm,
        annot=cm_norm,
        annot_kws={"size": 16},
        cmap="Blues",
        cbar=False,
        fmt=".2%" if cm_norm.min() < 0.01 else ".1%",
    )
    ax.set_xticks(ticks=[0.5, 1.5])
    ax.set_xticklabels(labels=["FP", "TP"])
    ax.set_xlabel("Predicted label")
    ax.set_yticks(ticks=[0.5, 1.5])
    ax.set_yticklabels(labels=["FP", "TP"], rotation="horizontal")
    ax.set_ylabel("True label")

    title_handle = plt.title(title)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )


def plot_observed_vs_measured_qual(
    labels_dict: dict,
    fprs: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and saves a plot of observed ML qual vs measured FP rates

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    fprs : dict
        list of false positive rates per label
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    plt.figure(figsize=(8, 6))
    for label in labels_dict:
        plot_precision_recall(
            fprs[label],
            [f"measured qual {labels_dict[label]}"],
            log_scale=False,
            max_score=max_score,
        )
    plt.plot([0, max_score], [0, max_score], "--")
    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("measured qual", fontsize=font_size)
    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_qual_density(
    labels_dict: dict,
    recalls: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and saves a plot of measured recall rates vs ML qual

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    recalls : dict
        list of recalls rates per label
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    plt.figure(figsize=(8, 6))

    for label in labels_dict:
        plot_precision_recall(
            recalls[label],
            [f"density {labels_dict[label]}"],
            log_scale=False,
            max_score=max_score,
        )

    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("qual density", fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_precision_recall_vs_qual_thresh(
    df: pd.DataFrame,
    labels_dict: dict,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and saves a plot of precision and recall rates vs ML qual threshold

    Parameters
    ----------
    df : pd.DataFrame
        data set with features, labels and quals of the model
    labels_dict : dict
        dict of label values and names
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    plt.figure(figsize=(8, 6))
    plt.title("precision/recall average as a function of min-qual")
    for label in labels_dict:
        cum_avg_precision_recalls = []
        gtr = df["label"] == label
        cum_fprs_, cum_recalls_ = precision_recall_curve(
            df[f"ML_qual_{label}"],
            max_score=max_score,
            y_true=gtr,
            cumulative=True,
            apply_log_trans=False,
        )
        cum_avg_precision_recalls.append(
            [(precision + recall) / 2 for precision, recall in zip(cum_fprs_, cum_recalls_)]
        )

        plot_precision_recall(
            cum_avg_precision_recalls,
            [f"avg(precision,recall) {labels_dict[label]}"],
            log_scale=False,
            max_score=max_score,
        )

    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    plt.xlabel("ML qual thresh", fontsize=font_size)
    plt.ylabel("precision/recall average", fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_ML_qual_hist(
    labels_dict: dict,
    df: pd.DataFrame,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and save histogram of ML qual per label

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    df : pd.DataFrame
        data set with features, labels and quals of the model
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    score = "ML_qual_1"

    plt.figure(figsize=[8, 6])
    plt.title("ML qual distribution per label")
    bins = np.arange(0, max_score + 1)
    for label in labels_dict:
        plt.hist(
            df[df["label"] == label][score].clip(upper=max_score),
            bins=bins,
            alpha=0.5,
            label=labels_dict[label],
            density=True,
        )

    plt.xlabel("ML qual", fontsize=font_size)
    plt.ylabel("Density", fontsize=font_size)
    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_qual_per_feature(
    labels_dict: dict,
    cls_features: list,
    df: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and save distributions of ML qual per input feature

    df : pd.DataFrame
        data set with features, labels and quals of the model
    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    cls_features : list
        list of input features for the ML model
    df : pd.DataFrame
        data set with features, labels and quals of the model
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    if "is_mixed" in df:
        df["is_mixed"] = df["is_mixed"].astype(int)
    for feature in cls_features:
        for j, label in enumerate(labels_dict):
            if df[feature].dtype == bool:
                if j == 0:
                    plt.figure(figsize=(6, 3))
                _ = (
                    df[df["label"] == label][feature]
                    .astype(int)
                    .hist(bins=20, alpha=0.5, label=labels_dict[label], density=True)
                )
            elif df[feature].dtype in {"category", "object"}:
                if j == 0:
                    plt.figure(figsize=(16, 4))
                _ = df[df["label"] == label][feature].hist(bins=20, alpha=0.5, label=labels_dict[label], density=True)
            else:
                if j == 0:
                    plt.figure(figsize=(8, 4))
                _ = df[df["label"] == label][feature].hist(bins=20, alpha=0.5, label=labels_dict[label], density=True)

        xticks = plt.gca().get_xticks()
        if len(xticks) > 100:
            plt.xticks(rotation=90, fontsize=6)
        elif len(xticks) > 30:
            plt.xticks(rotation=90, fontsize=9)

        legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
        plt.xlabel(feature)
        title_handle = plt.title(title, fontsize=font_size)
        output_filename_feature = output_filename + feature
        if output_filename_feature is not None:
            if not output_filename_feature.endswith(".png"):
                output_filename_feature += ".png"
        plt.savefig(
            output_filename_feature,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def get_mixed_data(
    df: pd.DataFrame,
):
    """generates subsets of the input df by category: mixed/non mixed * cycle-skip/non cycle-skip

    Parameters
    ----------
    df : pd.DataFrame
        data set with features, labels and quals of the model

    Returns
    -------
    df_mixed_cs: pd.DataFrame
        subset of input df: mixed and cycle-skip
    df_mixed_non_cs: pd.DataFrame
        subset of input df: mixed and non cycle-skip
    df_non_mixed_non_cs: pd.DataFrame
        subset of input df: non mixed and non cycle-skip
    df_non_mixed_cs: pd.DataFrame
        subset of input df: non mixed and cycle-skip
    """
    df_mixed_cs = df[(df["is_mixed"] & df["is_cycle_skip"])]
    df_mixed_non_cs = df[(df["is_mixed"] & ~df["is_cycle_skip"])]
    df_non_mixed_non_cs = df[(~df["is_mixed"] & ~df["is_cycle_skip"])]
    df_non_mixed_cs = df[(~df["is_mixed"] & df["is_cycle_skip"])]
    return df_mixed_cs, df_mixed_non_cs, df_non_mixed_non_cs, df_non_mixed_cs


def get_fpr_recalls_mixed(
    df_mixed_cs: pd.DataFrame,
    df_mixed_non_cs: pd.DataFrame,
    df_non_mixed_cs: pd.DataFrame,
    df_non_mixed_non_cs: pd.DataFrame,
    max_score: float,
):
    """get the FP and recall rates for subsamples of the data: mixed/non mixed * cycle skip/non cycle skip

    Parameters
    ----------
    df_mixed_cs : pd.DataFrame
        data subset with features, labels and quals of the model, mixed and cycle-skip
    df_mixed_non_cs : pd.DataFrame
        data subset with features, labels and quals of the model, mixed and non cycle-skip
    df_non_mixed_cs : pd.DataFrame
        data subset with features, labels and quals of the model, non mixed and cycle-skip
    df_non_mixed_non_cs : pd.DataFrame
        data subset with features, labels and quals of the model, non mixed and non cycle-skip
    max_score : float
        maximal ML model score

    Returns
    -------
    fprs_mixed_cs: list
        list of FP rates for mixed and cycle skip reads
    recalls_mixed_cs: list
        list of recall rates for mixed and cycle skip reads
    fprs_mixed_non_cs: list
        list of FP rates for mixed and non cycle skip reads
    recalls_mixed_non_cs: list
        list of recall rates for mixed and non cycle skip reads
    fprs_non_mixed_cs: list
        list of FP rates for non mixed and cycle skip reads
    recalls_non_mixed_cs: list
        list of recall rates for non mixed and cycle skip reads
    fprs_non_mixed_non_cs: list
        list of FP rates for non mixed and non cycle skip reads
    recalls_non_mixed_non_cs: list
        list of recall rates for non mixed and non cycle skip reads
    """
    score = "ML_qual_1"
    label = 1
    gtr = df_mixed_cs["label"] == label
    fprs_mixed_cs, recalls_mixed_cs = precision_recall_curve(df_mixed_cs[score], max_score=max_score, y_true=gtr)
    gtr = df_mixed_non_cs["label"] == label
    fprs_mixed_non_cs, recalls_mixed_non_cs = precision_recall_curve(
        df_mixed_non_cs[score], max_score=max_score, y_true=gtr
    )
    gtr = df_non_mixed_cs["label"] == label
    fprs_non_mixed_cs, recalls_non_mixed_cs = precision_recall_curve(
        df_non_mixed_cs[score], max_score=max_score, y_true=gtr
    )
    gtr = df_non_mixed_non_cs["label"] == label
    fprs_non_mixed_non_cs, recalls_non_mixed_non_cs = precision_recall_curve(
        df_non_mixed_non_cs[score], max_score=max_score, y_true=gtr
    )

    return (
        fprs_mixed_cs,
        recalls_mixed_cs,
        fprs_mixed_non_cs,
        recalls_mixed_non_cs,
        fprs_non_mixed_cs,
        recalls_non_mixed_cs,
        fprs_non_mixed_non_cs,
        recalls_non_mixed_non_cs,
    )


def plot_mixed(
    labels_dict: dict,
    df_mixed_cs: pd.DataFrame,
    df_mixed_non_cs: pd.DataFrame,
    df_non_mixed_non_cs: pd.DataFrame,
    df_non_mixed_cs: pd.DataFrame,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and save histograms of ML qual, separated by subsets: mixed/non mixed and cycle-skip/non cycle-skip

    Parameters
    ----------
    labels_dict : dict
        dict of label values and names
    df_mixed_cs : pd.DataFrame
        data subset with features, labels and quals of the model, mixed and cycle-skip
    df_mixed_non_cs : pd.DataFrame
        data subset with features, labels and quals of the model, mixed and non cycle-skip
    df_non_mixed_non_cs : pd.DataFrame
        data subset with features, labels and quals of the model, non mixed and non cycle-skip
    df_non_mixed_cs : pd.DataFrame
        data subset with features, labels and quals of the model, non mixed and cycle-skip
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    score = "ML_qual_1"
    bins = np.arange(0, max_score + 1)

    for td, name in zip(
        [df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs],
        ["mixed_cs", "mixed_non_cs", "non_mixed_cs", "non_mixed_non_cs"],
    ):
        plt.figure()
        plt.title(
            title + name + "\nMean ML_QUAL: {:.2f}, Median ML_QUAL: {:.2f}".format(td[score].mean(), td[score].median())
        )
        for label in labels_dict:
            _ = (
                td[td["label"] == label][score]
                .clip(upper=max_score)
                .hist(bins=bins, label=labels_dict[label], alpha=0.8, density=True)
            )
        plt.xlim([0, max_score])
        legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
        feature_title = title + name
        title_handle = plt.title(feature_title, fontsize=font_size)
        output_filename_feature = output_filename + name
        if output_filename_feature is not None:
            if not output_filename_feature.endswith(".png"):
                output_filename_feature += ".png"
        plt.savefig(
            output_filename_feature,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )


def plot_mixed_fpr(
    fprs_mixed_cs: list,
    fprs_mixed_non_cs: list,
    fprs_non_mixed_cs: list,
    fprs_non_mixed_non_cs: list,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and save a plot of FP rates per subset: mixed/non-mixed * cycle-skip/non cycle-skip

    Parameters
    ----------
    fprs_mixed_cs : list
        list of FP rates for mixed and cycle skip reads
    fprs_mixed_non_cs : list
        list of FP rates for mixed and non cycle skip reads
    fprs_non_mixed_cs : list
        list of FP rates for non mixed and cycle skip reads
    fprs_non_mixed_non_cs : list
        list of FP rates for non mixed and non cycle skip reads
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    plt.figure(figsize=(8, 6))
    plot_precision_recall(
        [fprs_mixed_cs, fprs_mixed_non_cs, fprs_non_mixed_cs, fprs_non_mixed_non_cs],
        [
            "fpr (mixed & cs)",
            "fpr (mixed & ~cs)",
            "fpr (~mixed & cs)",
            "fpr (~mixed & ~cs)",
        ],
        log_scale=False,
        max_score=max_score,
    )
    plt.plot([0, 40], [0, 40], "--")
    plt.xlim([0, max_score])
    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
    plt.savefig(
        output_filename,
        facecolor="w",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[title_handle, legend_handle],
    )


def plot_mixed_recall(
    recalls_mixed_cs: list,
    recalls_mixed_non_cs: list,
    recalls_non_mixed_cs: list,
    recalls_non_mixed_non_cs: list,
    max_score: float,
    title: str = "",
    output_filename: str = None,
    font_size: int = 14,
):
    """generate and save a plot of recall rates per subset: mixed/non-mixed * cycle-skip/non cycle-skip

    Parameters
    ----------
    recalls_mixed_cs : list
        list of recall rates for mixed and cycle skip reads
    recalls_mixed_non_cs : list
        list of recall rates for mixed and non cycle skip reads
    recalls_non_mixed_cs : list
        list of recall rates for non mixed and cycle skip reads
    recalls_non_mixed_non_cs : list
        list of recall rates for non mixed and non cycle skip reads
    max_score : float
        maximal ML model score
    title : str, optional
        title for the generated plot, by default ""
    output_filename : str, optional
        path to which the plot will be saved, by default None
    font_size : int, optional
        font size for the plot, by default 14
    """
    set_default_plt_rc_params()

    plt.figure(figsize=(8, 6))
    plot_precision_recall(
        [
            recalls_mixed_cs,
            recalls_mixed_non_cs,
            recalls_non_mixed_cs,
            recalls_non_mixed_non_cs,
        ],
        [
            "recalls (mixed & cs)",
            "recalls (mixed & ~cs)",
            "recalls (~mixed & cs)",
            "recalls (~mixed & ~cs)",
        ],
        log_scale=False,
        max_score=max_score,
    )

    plt.xlim([0, max_score])
    legend_handle = plt.legend(fontsize=font_size, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=font_size)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
    plt.savefig(
        output_filename,
        facecolor="w",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[title_handle, legend_handle],
    )


def _get_plot_paths(report_name, out_path, out_basename):
    outdir = out_path
    basename = out_basename

    output_roc_plot = os.path.join(outdir, f"{basename}{report_name}.ROC_curve")
    output_LoD_plot = os.path.join(outdir, f"{basename}{report_name}.LoD_curve")
    output_cm_plot = os.path.join(outdir, f"{basename}{report_name}.confusion_matrix")
    output_precision_recall_qual = os.path.join(outdir, f"{basename}{report_name}.precision_recall_qual")
    output_qual_density = os.path.join(outdir, f"{basename}{report_name}.qual_density")
    output_obsereved_qual_plot = os.path.join(outdir, f"{basename}{report_name}.observed_qual")
    output_ML_qual_hist = os.path.join(outdir, f"{basename}{report_name}.ML_qual_hist")
    output_qual_per_feature = os.path.join(outdir, f"{basename}{report_name}.qual_per_")
    output_balanced_strand_hists = os.path.join(outdir, f"{basename}{report_name}.balanced_strand_")
    output_balanced_strand_fpr = os.path.join(outdir, f"{basename}{report_name}.balanced_strand_fpr")
    output_balanced_strand_recalls = os.path.join(outdir, f"{basename}{report_name}.balanced_strand_recalls")

    return [
        output_roc_plot,
        output_LoD_plot,
        output_cm_plot,
        output_precision_recall_qual,
        output_qual_density,
        output_obsereved_qual_plot,
        output_ML_qual_hist,
        output_qual_per_feature,
        output_balanced_strand_hists,
        output_balanced_strand_fpr,
        output_balanced_strand_recalls,
    ]


def calculate_lod_stats(
    df_mrd_simulation: pd.DataFrame,
    output_h5: str,
    lod_column: str,
    qualities_to_interpolate: tuple | list | np.array = tuple(range(30, 71)),
):
    """Calculate noise and LoD stats from the simulated data

    Parameters
    ----------
    df_mrd_simulation : pd.DataFrame
        simulated data set with noise and LoD stats
    output_h5 : str
        path to output h5 file
    lod_column : str
        name of the LoD column in the data set
    qualities_to_interpolate : tuple
        list of qualities to interpolate what percent of bases surpass, by default all integers in [30,70]

    Returns
    -------
    best_LoD_filter: str
        name of the filter with the best LoD
    """
    df_mrd_simulation_ml = df_mrd_simulation.loc[df_mrd_simulation.index.str.startswith("ML")]
    df_mrd_simulation_non_ml = df_mrd_simulation.loc[~df_mrd_simulation.index.str.startswith("ML")]
    # Calculate percent of bases over each given quality threshold
    x_interp = -10 * np.log10(df_mrd_simulation_ml["residual_snv_rate"])  # Phred score
    y_interp = df_mrd_simulation_ml["tp_read_retention_ratio"]
    base_stats = pd.Series(
        index=qualities_to_interpolate,
        data=interp1d(x_interp, y_interp, bounds_error=False, fill_value=(1, 0))(qualities_to_interpolate),
    )  # fill_value=(1, 0) means that values under the minimal Qual will be 1 and values over the maximal will be 0
    base_stats.index.name = "QualThreshold"
    base_stats.name = "BasesOverQualThreshold"
    base_stats.to_hdf(output_h5, key="bases_over_qual_threshold", mode="a")

    # Calculate Optimal filter and LoD result
    best_LoD_filter = df_mrd_simulation[lod_column].idxmin()
    best_lod = df_mrd_simulation.loc[best_LoD_filter, lod_column]
    best_ML_LoD_filter = df_mrd_simulation_ml[lod_column].idxmin()
    best_ML_LoD = df_mrd_simulation_ml.loc[best_ML_LoD_filter, lod_column]
    lod_stats_dict = {
        "best_LoD_filter": best_LoD_filter,
        "LoD_best": best_lod,
        "best_ML_LoD_filter": best_ML_LoD_filter,
        "LoD_best_ML": best_ML_LoD,
    }
    for column, stat_name in zip(
        (lod_column, RESIDUAL_SNV_RATE, TP_READ_RETENTION_RATIO),
        ("LoD", RESIDUAL_SNV_RATE, TP_READ_RETENTION_RATIO),
    ):
        lod_stats_dict.update(
            {f"{stat_name}-{filter_name}": row[column] for filter_name, row in df_mrd_simulation_non_ml.iterrows()}
        )

    logger.info(f"min_LoD_filter {best_LoD_filter} (LoD={best_lod:.1e})")
    lod_stats = pd.Series(lod_stats_dict)
    lod_stats.to_hdf(output_h5, key="lod_stats", mode="a")

    return best_LoD_filter


def create_report_plots(
    model_file: str,
    X_file: str,
    y_file: str,
    params_file: str,
    report_name: str,
    out_path: str,
    base_name: str = None,
    lod_filters: dict = None,
    mrd_simulation_dataframe_file: str = None,
    statistics_h5_file: str = None,
    statistics_json_file: str = None,
):
    """loads model, data, params and generate plots for report

    Parameters
    ----------
    model_file : str
        path to model file
    X_file : str
        path to data
    y_file : str
        path to labels
    params_file : str
        path to params file
    report_name : str
        name of data set, should be "train" or "test"
    out_path : str
        path to output directory
    base_name : str, optional
        base name for output files, by default None
    lod_filters : dict, optional
        filters for LoD simulation, by default None (default_LoD_filters)
    mrd_simulation_dataframe_file : str, optional
        path to output simulated data set, by default None
    statistics_h5_file : str, optional
        path to output h5 file with stats, by default None
    statistics_json_file : str, optional
        path to output json file with stats, by default None
    """
    if statistics_json_file:
        assert statistics_h5_file, "statistics_h5_file is required when statistics_json_file is provided"

    # TODO: change the report so it produces metrics that are saved into a h5 table in addition to the plots

    # load model, data and params
    classifier = joblib.load(model_file)
    X = pd.read_parquet(X_file)
    y = pd.read_parquet(y_file)
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    params["workdir"] = out_path
    if base_name:
        params["data_name"] = base_name
    else:
        params["data_name"] = ""

    assert "sorter_json_stats_file" in params, "no sorter_json_stats_file in params"

    (
        df,
        df_tp,
        df_fp,
        max_score,
        cls_features,
        fprs,
        recalls,
    ) = create_data_for_report(classifier, X, y)

    labels_dict = {0: "FP", 1: "TP"}

    lod_basic_filters = lod_filters or default_LoD_filters
    ML_filters = {f"ML_qual_{q}": f"ML_qual_1 >= {q}" for q in range(0, max_score + 1)}

    lod_filters = {
        **lod_basic_filters,
        **ML_filters,
    }

    sorter_json_stats_file = params["sorter_json_stats_file"]
    data_size = params[f"{report_name}_set_size"]
    total_n = params["fp_featuremap_entry_number"]
    sampling_rate = (
        data_size / total_n
    )  # N reads in test (test_size) / N reads in single sub featuremap intersected with single_sub_regions

    [
        output_roc_plot,
        output_LoD_plot,
        output_cm_plot,
        output_precision_recall_qual,
        output_qual_density,
        output_obsereved_qual_plot,
        output_ML_qual_hist,
        output_qual_per_feature,
        output_balanced_strand_hists,
        output_balanced_strand_fpr,
        output_balanced_strand_recalls,
    ] = _get_plot_paths(report_name, out_path=params["workdir"], out_basename=params["data_name"])

    LoD_params = {}
    LoD_params["sensitivity_at_lod"] = 0.90
    LoD_params["specificity_at_lod"] = 0.99
    LoD_params["simulated_signature_size"] = 10_000
    LoD_params["simulated_coverage"] = 30
    LoD_params["minimum_number_of_read_for_detection"] = 2

    if params["fp_regions_bed_file"] is not None:
        (df_mrd_simulation, lod_filters_filtered, lod_label, c_lod,) = retention_noise_and_mrd_lod_simulation(
            df=df,
            single_sub_regions=params["fp_regions_bed_file"],
            sorter_json_stats_file=sorter_json_stats_file,
            training_set_downsampling_rate=sampling_rate,
            lod_filters=lod_filters,
            sensitivity_at_lod=LoD_params["sensitivity_at_lod"],
            specificity_at_lod=LoD_params["specificity_at_lod"],
            simulated_signature_size=LoD_params["simulated_signature_size"],
            simulated_coverage=LoD_params["simulated_coverage"],
            minimum_number_of_read_for_detection=LoD_params["minimum_number_of_read_for_detection"],
            output_dataframe_file=mrd_simulation_dataframe_file,
        )
        min_LoD_filter = calculate_lod_stats(
            df_mrd_simulation=df_mrd_simulation,
            output_h5=statistics_h5_file,
            lod_column=c_lod,
        )
        plot_LoD(
            df_mrd_simulation,
            lod_label,
            c_lod,
            lod_filters_filtered,
            params["adapter_version"],
            min_LoD_filter,
            title=f"{params['data_name']}\nLoD curve",
            output_filename=output_LoD_plot,
        )

    plot_ROC_curve(
        df,
        df_tp,
        df_fp,
        ML_score="ML_qual_1",
        adapter_version=params["adapter_version"],
        title=f"{params['data_name']}\nROC curve",
        output_filename=output_roc_plot,
    )

    plot_confusion_matrix(
        df,
        title=f"{params['data_name']}\nConfusion matrix",
        prediction_column_name="ML_prediction_1",
        output_filename=output_cm_plot,
    )
    plot_precision_recall_vs_qual_thresh(
        df,
        labels_dict,
        max_score,
        title=f"{params['data_name']}\nprecision/recall average as a function of min-qual",
        output_filename=output_precision_recall_qual,
    )
    plot_qual_density(
        labels_dict,
        recalls,
        max_score,
        title=f"{params['data_name']}\nmeasured recall rates vs ML qual",
        output_filename=output_qual_density,
    )
    plot_observed_vs_measured_qual(
        labels_dict,
        fprs,
        max_score,
        title=f"{params['data_name']}\nobserved qual vs. measured qual",
        output_filename=output_obsereved_qual_plot,
    )
    plot_ML_qual_hist(
        labels_dict,
        df,
        max_score,
        title=f"{params['data_name']}\nML qual distribution",
        output_filename=output_ML_qual_hist,
    )
    plot_qual_per_feature(
        labels_dict,
        cls_features,
        df,
        title=f"{params['data_name']}",
        output_filename=output_qual_per_feature,
    )

    if "is_mixed" in df:
        (
            df_mixed_cs,
            df_mixed_non_cs,
            df_non_mixed_non_cs,
            df_non_mixed_cs,
        ) = get_mixed_data(df)
        (
            fprs_mixed_cs,
            recalls_mixed_cs,
            fprs_mixed_non_cs,
            recalls_mixed_non_cs,
            fprs_non_mixed_cs,
            recalls_non_mixed_cs,
            fprs_non_mixed_non_cs,
            recalls_non_mixed_non_cs,
        ) = get_fpr_recalls_mixed(
            df_mixed_cs,
            df_mixed_non_cs,
            df_non_mixed_cs,
            df_non_mixed_non_cs,
            max_score,
        )

        plot_mixed(
            labels_dict,
            df_mixed_cs,
            df_mixed_non_cs,
            df_non_mixed_non_cs,
            df_non_mixed_cs,
            max_score,
            title=f"{params['data_name']}\nbalanced_strand: ",
            output_filename=output_balanced_strand_hists,
        )

        plot_mixed_fpr(
            fprs_mixed_cs,
            fprs_mixed_non_cs,
            fprs_non_mixed_cs,
            fprs_non_mixed_non_cs,
            max_score,
            title=f"{params['data_name']}\nbalanced_strand fp rate vs. qual ",
            output_filename=output_balanced_strand_fpr,
        )
        plot_mixed_recall(
            recalls_mixed_cs,
            recalls_mixed_non_cs,
            recalls_non_mixed_cs,
            recalls_non_mixed_non_cs,
            max_score,
            title=f"{params['data_name']}\nbalanced_strand recalls vs. qual ",
            output_filename=output_balanced_strand_recalls,
        )

    # convert statistics to json
    convert_h5_to_json(
        input_h5_filename=statistics_h5_file,
        root_element="metrics",
        ignored_h5_key_substring=None,
        output_json=statistics_json_file,
    )


def precision_score_with_mask(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    """calculates the precision score for the predictions and labels with mask applied

    Parameters
    ----------
    y_pred : np.ndarray
        model predictions
    y_true : np.ndarray
        labels
    mask : np.ndarray
        mask for the data

    Returns
    -------
    float
        sklearn precision score for the predictions and labels with mask applied
    """
    if y_pred[mask].sum() == 0:
        return 1
    return precision_score(y_true[mask], y_pred[mask])


def recall_score_with_mask(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    """calculates the recall score for the predictions and labels with mask applied

    Parameters
    ----------
    y_pred : np.ndarray
        model predictions
    y_true : np.ndarray
        labels
    mask : np.ndarray
        mask for the data

    Returns
    -------
    float
        sklearn recall score for the predictions and labels with mask applied
    """
    if y_true[mask].sum() == 0:
        return 1
    return recall_score(y_true[mask], y_pred[mask])


def precision_recall_curve(score, max_score, y_true: np.ndarray, cumulative=False, apply_log_trans=True):
    """apply threshold on score and calculate precision and recall rates

    Parameters
    ----------
    score : pd.dataframe
        model score
    max_score : _type_
        maximal ML model score for threshold
    y_true : np.ndarray
        labels
    cumulative : bool, optional
        whether to calculate cumulative scores, by default False
    apply_log_trans : bool, optional
        whether to use precision for false positive rate, or apply log transformation to get qual, by default True

    Returns
    -------
    fprs : list
        list of false positive rates per threshold
    recalls : list
        list of recall rates per threshold
    """
    precisions = []
    recalls = []
    fprs = []
    for i in range(max_score):
        if cumulative:
            mask = score.apply(np.floor) >= i
        else:
            mask = score.apply(np.floor) == i
        prediction = mask
        no_mask = (score + 1).astype(bool)
        precisions.append(precision_score_with_mask(prediction, y_true, mask))
        recalls.append(recall_score_with_mask(prediction, y_true, no_mask))
        if precisions[-1] == np.nan:
            fprs.append(np.none)
        elif apply_log_trans:
            if precisions[-1] == 1:
                qual = max_score
            else:
                qual = -10 * np.log10(1 - precisions[-1])
            qual = min(qual, max_score)
            fprs.append(qual)
        else:
            fprs.append(precisions[-1])
    return fprs, recalls


def plot_precision_recall(lists, labels, max_score, log_scale=False, font_size=22):
    """generate a plot of precision and recall rates per threshold

    Parameters
    ----------
    lists : list[list]
        lists of precision and recall rates per threshold
    labels : list[str]
        list of labels for the lists
    max_score : float
        maximal model score for plot
    log_scale : bool, optional
        whether to plot in log scale, by default False
    font_size : int, optional
        font size, by default 14
    """
    set_default_plt_rc_params()

    for lst, label in zip(lists, labels):
        plt.plot(lst[0:max_score], ".-", label=label)
        if log_scale:
            plt.yscale("log")
        plt.xlabel("QUAL", fontsize=font_size)
