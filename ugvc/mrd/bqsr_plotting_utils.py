from __future__ import annotations

import json
import os
from os.path import join as pjoin

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy.stats import binom
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ugvc.mrd.bqsr_train_utils import inference_on_dataframe
from ugvc.srsnv.srsnv_utils import plot_precision_recall, precision_recall_curve
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json
from ugvc.utils.misc_utils import exec_command_list


def bqsr_train_report(
    out_path,
    out_basename,
    report_name,
    model_path,
    X_path,
    y_path,
    fp_bed_file,
    params_path,
    simple_pipeline=None,
):
    reportfile = pjoin(out_path, f"{out_basename}{report_name}_report.ipynb")
    reporthtml = pjoin(out_path, f"{out_basename}{report_name}_report.html")

    commands = [
        f"papermill ugvc/reports/bqsr_train_report.ipynb {reportfile} -p report_name {report_name} \
            -p model_file {model_path} -p X_file {X_path} -p y_file {y_path} \
            -p single_sub_regions {fp_bed_file} -p params_file {params_path} -k python3",
        f"jupyter nbconvert {reportfile} --output {reporthtml} --to html --template classic --no-input",
    ]
    exec_command_list(commands, simple_pipeline)


def plot_ROC_curve(
    df: pd.DataFrame,
    df_tp: pd.DataFrame,
    df_fp: pd.DataFrame,
    score: str = "",
    title: str = "",
    output_filename: str = None,
):
    label_dict = {
        "X_SCORE": "FeatureMap",
        "X_SCORE_mixed": "FeatureMap",
        score: "xgb model",
    }

    recall = {}
    precision = {}
    for xvar, querystr, namestr in (
        ("X_SCORE", "X_SCORE>-1", "X_SCORE"),
        ("X_SCORE", "is_mixed == True & X_SCORE>-1", "X_SCORE_mixed"),
        (score, "X_SCORE>-1", score),
    ):
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


def LoD_training_summary(
    single_sub_regions, cram_stats_file, sampling_rate, df, filters_file, df_summary_file, output_params_file
):

    # apply filters to data
    with open(filters_file, encoding="utf-8") as f:
        filters = json.load(f)

    df = df.assign(
        **{
            filter_name: df.eval(filter_query)
            for filter_name, filter_query in tqdm(filters.items())
            if filter_query is not None
        }
    )
    df_fp = df.query("label == 0")
    df_tp = df.query("label == 1")

    # count the # of bases in region
    n_bases_in_region = 0
    with open(single_sub_regions, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("@") and not line.startswith("#"):
                spl = line.rstrip().split("\t")
                n_bases_in_region += int(spl[2]) - int(spl[1])

    # get coverage statistics
    (
        mean_coverage,
        ratio_of_reads_over_mapq,
        ratio_of_bases_in_coverage_range,
        _,
        _,
    ) = read_effective_coverage_from_sorter_json(cram_stats_file)

    read_filter_correction_factor = (df_tp["X_FILTERED_COUNT"]).sum() / df_tp["X_READ_COUNT"].sum()
    n_noise_reads = df_fp.shape[0]
    n_signal_reads = df_tp.shape[0]
    print("n_noise_reads, n_signal_reads", n_noise_reads, n_signal_reads)
    effective_bases_covered = (
        mean_coverage
        * n_bases_in_region
        * sampling_rate
        * ratio_of_reads_over_mapq
        * ratio_of_bases_in_coverage_range
        * read_filter_correction_factor
    )
    residual_snv_rate_no_filter = n_noise_reads / effective_bases_covered
    ratio_filtered_prior_to_featuremap = ratio_of_reads_over_mapq * read_filter_correction_factor

    df_summary = pd.concat(
        (
            (df_fp[list(filters.keys())].sum() / n_noise_reads).rename("FP"),
            (df_tp[list(filters.keys())].sum() / n_signal_reads * ratio_filtered_prior_to_featuremap).rename("TP"),
        ),
        axis=1,
    )

    df_summary = df_summary.assign(precision=df_summary["TP"] / (df_summary["FP"] + df_summary["TP"]))

    # save outputs
    params_for_save = {"residual_snv_rate_no_filter": residual_snv_rate_no_filter}
    with open(output_params_file, "w", encoding="utf-8") as f:
        json.dump(params_for_save, f)

    df_summary.to_parquet(df_summary_file)

    return filters


def LoD_simulation_on_signature(LoD_params):

    df_summary = pd.read_parquet(LoD_params["df_summary_file"])

    with open(LoD_params["summary_params_file"], "r", encoding="utf-8") as f:
        summary_params = json.load(f)

    residual_snv_rate_no_filter = summary_params["residual_snv_rate_no_filter"]

    sensitivity_at_lod = LoD_params["sensitivity_at_lod"]  # 0.90
    specificity_at_lod = LoD_params["specificity_at_lod"]  # 0.99

    # simulated LoD definitions and correction factors
    simulated_signature_size = LoD_params["simulated_signature_size"]  # 10_000
    simulated_coverage = LoD_params["simulated_coverage"]  # 30
    effective_signature_bases_covered = (
        simulated_coverage * simulated_signature_size
    )  # * ratio_of_reads_over_mapq * read_filter_correction_factor

    df_mrd_sim = (df_summary[["TP"]]).rename(columns={"TP": "read_retention_ratio"})
    df_mrd_sim = df_mrd_sim.assign(
        residual_snv_rate=df_summary["FP"] * residual_snv_rate_no_filter / df_mrd_sim["read_retention_ratio"]
    )
    df_mrd_sim = df_mrd_sim.assign(
        min_reads_for_detection=np.ceil(
            binom.ppf(
                n=int(effective_signature_bases_covered),
                p=df_mrd_sim["read_retention_ratio"] * df_mrd_sim["residual_snv_rate"],
                q=specificity_at_lod,
            )
        )
        .clip(min=2)
        .astype(int),
    )

    tf_sim = np.logspace(-8, 0, 500)

    df_mrd_sim = df_mrd_sim.join(
        df_mrd_sim.apply(
            lambda row: tf_sim[
                np.argmax(
                    binom.ppf(
                        q=1 - sensitivity_at_lod,
                        n=int(effective_signature_bases_covered),
                        p=row["read_retention_ratio"] * (tf_sim + row["residual_snv_rate"]),
                    )
                    >= row["min_reads_for_detection"]
                )
            ],
            axis=1,
        ).rename(f"LoD_{sensitivity_at_lod*100:.0f}")
    )

    df_mrd_sim = df_mrd_sim[df_mrd_sim["read_retention_ratio"] > 0.01]
    lod_label = f"LoD @ {specificity_at_lod*100:.0f}% specificity, \
                        {sensitivity_at_lod*100:.0f}% sensitivity (estimated)\
                        \nsignature size {simulated_signature_size}, \
                        {simulated_coverage}x coverage"
    c_lod = f"LoD_{sensitivity_at_lod*100:.0f}"
    min_LoD_filter = df_mrd_sim["LoD_90"].idxmin()

    return df_mrd_sim, lod_label, c_lod, min_LoD_filter


def plot_LoD(df_mrd_sim, lod_label, c_lod, filters, title="", output_filename="", fs=14):
    xgb_filters = {i: filters[i] for i in filters if i[:3] == "xgb" and i in df_mrd_sim.index}

    plt.figure(figsize=(20, 12))

    for df_tmp, marker, label, edgecolor, markersize in zip(
        (
            df_mrd_sim.loc[["no_filter"]],
            df_mrd_sim.loc[["BQ80"]],
            df_mrd_sim.loc[["BQ80_mixed_only"]],
            df_mrd_sim.loc[list(xgb_filters)],
        ),
        (
            "*",
            "D",
            "<",
            ">",
            "s",
        ),
        (
            "No filter",
            "BQ80 and Edit Dist <= 5",
            "BQ80 and Edit Dist <= 5 (mixed only)",
            "xgb",
        ),
        ("r", "r", "r", "r", "none", "none"),
        (150, 150, 150, 150, 100, 100),
    ):

        plt.plot(
            df_tmp["read_retention_ratio"],
            df_tmp["residual_snv_rate"],
            c="k",
            alpha=0.3,
        )
        best_lod = df_tmp[c_lod].min()
        plt.scatter(
            df_tmp["read_retention_ratio"],
            df_tmp["residual_snv_rate"],
            c=df_tmp[c_lod],
            marker=marker,
            edgecolor=edgecolor,
            label=label + ", best LoD: {:.1E}".format(best_lod).replace("E-0", "E-"),
            s=markersize,
            zorder=markersize,
        )

    plt.xlabel("Read retention ratio on HOM SNVs", fontsize=fs)
    plt.ylabel("Residual SNV rate", fontsize=fs)
    plt.yscale("log")
    title_handle = plt.title(title, fontsize=fs)
    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)

    def fmt(x, pos):
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        return r"{}${} \times 10^{{{}}}$".format(pos, a, b)

    cb = plt.colorbar(format=ticker.FuncFormatter(fmt))
    cb.set_label(label=lod_label, fontsize=20)
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
    fs: int = 14,
):
    cm = confusion_matrix(df["XGB_prediction_1"], df["label"])
    cm_norm = cm / cm.sum(axis=0)
    plt.figure(figsize=(4, 3))
    ax = sns.heatmap(cm_norm, annot=cm_norm, annot_kws={"size": fs})
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)

    title_handle = plt.title(title, fontsize=fs)

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
    labels_dict,
    fprs,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
    plt.figure(figsize=(8, 6))
    for label in labels_dict:
        plot_precision_recall(
            fprs[label],
            [f"measured qual {labels_dict[label]}"],
            log_scale=False,
            max_score=max_score,
        )
    plt.plot([0, max_score], [0, max_score], "--")
    plt.xlabel("ML qual", fontsize=fs)
    plt.ylabel("measured qual", fontsize=fs)
    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=fs)

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
    labels_dict,
    recalls,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
    plt.figure(figsize=(8, 6))

    for label in labels_dict:
        plot_precision_recall(
            recalls[label],
            [f"density {labels_dict[label]}"],
            log_scale=False,
            max_score=max_score,
        )

    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=fs)
    plt.xlabel("ML qual", fontsize=fs)
    plt.ylabel("qual density", fontsize=fs)

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
    df,
    labels_dict,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
    plt.figure(figsize=(8, 6))
    plt.title("precision/recall average as a function of min-qual")
    for label in labels_dict:
        cum_avg_precision_recalls = []
        gtr = df["label"] == label
        cum_fprs_, cum_recalls_ = precision_recall_curve(
            df[f"XGB_qual_{label}"],
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

    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=fs)
    plt.xlabel("ML qual thresh", fontsize=fs)
    plt.ylabel("precision/recall average", fontsize=fs)

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
    labels_dict,
    df,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs: int = 14,
):
    score = "XGB_qual_1"

    plt.figure(figsize=[8, 6])
    plt.title("xgb")
    bins = np.linspace(0, max_score, 50)
    for label in labels_dict:
        plt.hist(
            df[df["label"] == label][score].clip(upper=max_score),
            bins=bins,
            alpha=0.5,
            label=labels_dict[label],
            density=True,
        )

    plt.xlabel("ML qual", fontsize=fs)
    plt.ylabel("Density", fontsize=fs)
    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=fs)

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
    labels_dict,
    cls_features,
    df,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
    features = cls_features
    if "is_mixed" in df:
        df["is_mixed"] = df["is_mixed"].astype(int)
    for feature in features:
        plt.figure()
        for label in labels_dict:
            _ = df[df["label"] == label][feature].hist(bins=20, alpha=0.5, label=labels_dict[label], density=True)
        legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
        feature_title = title + feature
        title_handle = plt.title(feature_title, fontsize=fs)
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


def get_mixed_data(df):
    df_mixed_cs = df[(df["is_mixed"]) & (df["cycle_skip_status"] == "cycle-skip")]
    df_mixed_non_cs = df[(df["is_mixed"]) & (df["cycle_skip_status"] != "cycle-skip")]
    df_non_mixed_non_cs = df[(~df["is_mixed"]) & (df["cycle_skip_status"] != "cycle-skip")]
    df_non_mixed_cs = df[(~df["is_mixed"]) & (df["cycle_skip_status"] == "cycle-skip")]
    return df_mixed_cs, df_mixed_non_cs, df_non_mixed_non_cs, df_non_mixed_cs


def get_fpr_recalls_mixed(df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs, max_score):
    score = "XGB_qual_1"
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
    labels_dict,
    df_mixed_cs,
    df_mixed_non_cs,
    df_non_mixed_non_cs,
    df_non_mixed_cs,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs: int = 14,
):
    score = "XGB_qual_1"

    for td, name in zip(
        [df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs],
        ["mixed & cs", "mixed & ~cs", "~mixed & cs", "~mixed & ~cs"],
    ):
        # Mean and median ML_QUAL in [mixed/non-mixed]*[cskp/non-cskp]
        print(
            name,
            ": Mean ML_QUAL: {:.2f}, Median ML_QUAL: {:.2f}".format(td[score].mean(), td[score].median()),
        )

    for td, name in zip(
        [df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs],
        ["mixed_cs", "mixed_non_cs", "non_mixed_cs", "non_mixed_non_cs"],
    ):
        plt.figure()
        plt.title(title + name)
        for label in labels_dict:
            _ = td[td["label"] == label][score].clip(upper=max_score).hist(bins=20, label=labels_dict[label])
        plt.xlim([0, max_score])
        legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
        feature_title = title + name
        title_handle = plt.title(feature_title, fontsize=fs)
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
    fprs_mixed_cs,
    fprs_mixed_non_cs,
    fprs_non_mixed_cs,
    fprs_non_mixed_non_cs,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
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
    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=fs)
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
    recalls_mixed_cs,
    recalls_mixed_non_cs,
    recalls_non_mixed_cs,
    recalls_non_mixed_non_cs,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
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
    legend_handle = plt.legend(fontsize=fs, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=fs)
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


def create_report_plots(
    model_file,
    X_file,
    y_file,
    params_file,
    report_name,
):
    # load model, data and params
    xgb_classifier = joblib.load(model_file)
    X_test = pd.read_parquet(X_file)
    y_test = pd.read_parquet(y_file)
    params = None
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    (
        df,
        df_tp,
        df_fp,
        _,
        max_score,
        cls_features,
        fprs,
        recalls,
    ) = inference_on_dataframe(xgb_classifier, X_test, y_test)

    labels_dict = {0: "FP", 1: "TP"}
    cram_stats_file = params["cram_stats_file"]
    filters_file = "/data1/work/rinas/xgbpipeline/filters.json"
    data_size = params["model_params"][f"{report_name}_size"]
    total_n = params["total_n_single_sub_featuremap"]
    sampling_rate = (
        data_size / total_n
    )  # N reads in test (test_size) / N reads in single sub featuremap intersected with single_sub_regions
    # df_mrd_sim, lod_label, c_lod, filters = simulate_LoD(
    #     params["single_sub_regions"], cram_stats_file, sampling_rate, df, filters_file
    # )

    LoD_params = {}
    LoD_params["df_summary_file"] = pjoin(params["workdir"], "df_summary.parquet")
    LoD_params["summary_params_file"] = pjoin(params["workdir"], "summary_params.json")
    LoD_params["sensitivity_at_lod"] = 0.90
    LoD_params["specificity_at_lod"] = 0.99
    LoD_params["simulated_signature_size"] = 10_000
    LoD_params["simulated_coverage"] = 30

    filters = LoD_training_summary(
        params["single_sub_regions"],
        cram_stats_file,
        sampling_rate,
        df,
        filters_file,
        LoD_params["df_summary_file"],
        LoD_params["summary_params_file"],
    )
    df_mrd_sim, lod_label, c_lod, min_LoD_filter = LoD_simulation_on_signature(LoD_params)
    print(min_LoD_filter)

    output_roc_plot = os.path.join(params["workdir"], f"{params['out_basename']}ROC_curve")
    output_LoD_plot = os.path.join(params["workdir"], f"{params['out_basename']}LoD_curve")
    output_cm_plot = os.path.join(params["workdir"], f"{params['out_basename']}confusion_matrix")
    output_precision_recall_qual = os.path.join(params["workdir"], f"{params['out_basename']}precision_recall_qual")
    output_qual_density = os.path.join(params["workdir"], f"{params['out_basename']}qual_density")
    output_obsereved_qual_plot = os.path.join(params["workdir"], f"{params['out_basename']}observed_qual")
    output_ML_qual_hist = os.path.join(params["workdir"], f"{params['out_basename']}ML_qual_hist")
    output_qual_per_feature = os.path.join(params["workdir"], f"{params['out_basename']}qual_per_")
    output_bepcr_hists = os.path.join(params["workdir"], f"{params['out_basename']}bepcr_")

    plot_ROC_curve(
        df,
        df_tp,
        df_fp,
        score="XGB_qual_1",
        title=f"{params['data_name']}\nROC curve",
        output_filename=output_roc_plot,
    )
    plot_LoD(
        df_mrd_sim,
        lod_label,
        c_lod,
        filters,
        title=f"{params['data_name']}\nLoD curve",
        output_filename=output_LoD_plot,
    )
    plot_confusion_matrix(
        df,
        title=f"{params['data_name']}\nconfusion matrix",
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
        title=f"{params['data_name']}\nqual density",
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
        title=f"{params['data_name']}\nqual per ",
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
        ) = get_fpr_recalls_mixed(df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs, max_score)

        plot_mixed(
            labels_dict,
            df_mixed_cs,
            df_mixed_non_cs,
            df_non_mixed_non_cs,
            df_non_mixed_cs,
            max_score,
            title=f"{params['data_name']}\nbepcr: ",
            output_filename=output_bepcr_hists,
        )

        output_bepcr_fpr = os.path.join(params["workdir"], f"{params['out_basename']}bepcr_fpr")
        output_bepcr_recalls = os.path.join(params["workdir"], f"{params['out_basename']}bepcr_recalls")
        plot_mixed_fpr(
            fprs_mixed_cs,
            fprs_mixed_non_cs,
            fprs_non_mixed_cs,
            fprs_non_mixed_non_cs,
            max_score,
            title=f"{params['data_name']}\nbepcr fp rate vs. qual ",
            output_filename=output_bepcr_fpr,
        )
        plot_mixed_recall(
            recalls_mixed_cs,
            recalls_mixed_non_cs,
            recalls_non_mixed_cs,
            recalls_non_mixed_non_cs,
            max_score,
            title=f"{params['data_name']}\nbepcr recalls vs. qual ",
            output_filename=output_bepcr_recalls,
        )