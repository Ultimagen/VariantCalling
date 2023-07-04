from __future__ import annotations

import math
import warnings
from configparser import ConfigParser
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.ticker import FormatStrFormatter

from ugvc.utils.stats_utils import get_f1, get_precision, get_recall


def parse_config(config_file):
    parser = ConfigParser()
    parser.read(config_file)
    param_names = ["run_id", "pipeline_version", "h5_concordance_file"]
    parameters = {p: parser.get("VarReport", p) for p in param_names}
    parameters["verbosity"] = parser.get("VarReport", "verbosity", fallback="5")
    param_names.append("verbosity")

    # Optional parameters
    parameters["reference_version"] = parser.get("VarReport", "reference_version", fallback="hg38")
    parameters["truth_sample_name"] = parser.get("VarReport", "truth_sample_name", fallback="NA")
    parameters["h5outfile"] = parser.get("VarReport", "h5_output", fallback="var_report.h5")
    parameters["trained_w_gt"] = parser.get("VarReport", "h5_model_file", fallback=None)

    if parameters["truth_sample_name"]:
        param_names.append("truth_sample_name")

    for opt_param_name in (
        "model_name_with_gt",
        "model_name_without_gt",
        "model_pkl_with_gt",
        "model_pkl_without_gt",
        "model_name",
    ):
        opt_param = parser.get("VarReport", opt_param_name, fallback=None)
        if opt_param:
            parameters[opt_param_name] = opt_param
            param_names.append(opt_param_name)

    return parameters, param_names


class ErrorType(Enum):
    NOISE = 1
    NO_VARIANT = 2
    HOM_TO_HET = 3
    HET_TO_HOM = 4
    WRONG_ALLELE = 5
    NO_ERROR = 6


class ReportUtils:
    def __init__(self, verbosity, h5outfile: str, num_plots_in_row=6, min_value=0.2):
        self.verbosity = verbosity
        self.h5outfile = h5outfile
        self.min_value = min_value
        self.num_plots_in_row = num_plots_in_row
        self.score_name = "tree_score"

    def basic_analysis(self, data: pd.DataFrame, categories, out_key, out_key_sec=None):
        data_sec = None
        if out_key_sec is not None:
            sec_df = data.copy()
            if "blacklst" in sec_df.columns:
                is_sec = sec_df["blacklst"].apply(self.has_sec)
                sec_df.loc[is_sec, "filter"] = "SEC"
                sec_df.loc[is_sec & (sec_df["classify_gt"] == "tp"), "classify_gt"] = "fn"
                data_sec = sec_df[~(is_sec & (sec_df["classify_gt"] == "fp"))]

        opt_tab, opt_res, perf_curve, error_types_tab = self.get_performance(data, categories)

        if data_sec is not None:
            sec_opt_tab, sec_opt_res, _, sec_error_types_tab = self.get_performance(data_sec, categories)
            sec_opt_tab.to_hdf(self.h5outfile, key=out_key_sec)
            sec_error_types_tab.to_hdf(self.h5outfile, key=f"{out_key_sec}_error_types")

            sec_opt_tab = pd.concat([opt_tab, sec_opt_tab], keys=[out_key, "After filtering systematic errors"], axis=1)
            self.__display_tables(
                sec_opt_tab,
                error_types_tab,
                out_key,
                sec_error_types_tab,
            )
            if self.verbosity > 1:
                self.plot_performance(
                    perf_curve,
                    opt_res,
                    categories,
                    out_key,
                    opt_res_sec=sec_opt_res,
                )
        else:
            self.__display_tables(opt_tab, error_types_tab, out_key)
            if self.verbosity > 1:
                self.plot_performance(perf_curve, opt_res, categories, out_key)

        self.make_multi_index(opt_tab)
        opt_tab.to_hdf(self.h5outfile, key=out_key)
        error_types_tab.to_hdf(self.h5outfile, key=f"{out_key}_error_types")

    def homozygous_genotyping_analysis(self, d: pd.DataFrame, categories: list[str], out_key: str) -> None:
        hmz_data = d[(d["gt_ground_truth"] == (1, 1)) & (d["classify"] != "fn")]
        opt_tab, _, _, _ = self.get_performance(hmz_data, categories)
        display(opt_tab)
        self.make_multi_index(opt_tab)
        opt_tab.to_hdf(self.h5outfile, out_key)

    def base_stratification_analysis(self, d: pd.DataFrame, categories: list[str], bases: tuple) -> pd.DataFrame:
        base_data = d[
            (~d["indel"] & ((d["ref"] == bases[0]) | (d["ref"] == bases[1])))
            | ((d["hmer_length"] > 0) & ((d["hmer_indel_nuc"] == bases[0]) | (d["hmer_indel_nuc"] == bases[1])))
        ]

        opt_tab, opt_res, perf_curve, _ = self.get_performance(base_data, categories)
        opt_tab.rename(index={a: "{0} ({1}/{2})".format(a, bases[0], bases[1]) for a in opt_tab.index}, inplace=True)
        display(opt_tab)
        if self.verbosity > 1:
            self.plot_performance(perf_curve, opt_res, categories, str(bases))
        return opt_tab

    def plot_performance(
        self,
        perf_curve: dict,
        opt_res: dict,
        categories: list[str],
        name: str,
        opt_res_sec=None,
    ):
        m = self.num_plots_in_row
        for i in range(4, 10):
            drop_cat = f"hmer Indel {i}"
            if drop_cat in categories:
                categories.remove(drop_cat)
        n = math.ceil(len(categories) / m)
        num_empty_subplots = n * m - len(categories)

        fig_pr, ax_pr = plt.subplots(n, m, figsize=(3 * m, 3 * n + 0.5 * (n - 1)))
        fig_score, ax_score = plt.subplots(n, m, figsize=(3 * m, 3 * n + 0.5 * (n - 1)))

        opt_sec, ax_pr_row, ax_score_row = None, None, None
        for cat_index, cat in enumerate(categories):
            i = math.floor(cat_index / m)
            j = cat_index % m
            if len(ax_pr.shape) == 2:
                ax_pr_row = ax_pr[i]
                ax_score_row = ax_score[i]
            else:
                ax_pr_row = ax_pr
                ax_score_row = ax_score
            ax_pr_row[0].set_ylabel("Precision")

            perf = perf_curve[cat]

            opt = opt_res[cat]
            if opt_res_sec is not None:
                opt_sec = opt_res_sec[cat]
            if not perf.empty and not np.all(pd.isnull(perf.precision)) and not np.all(pd.isnull(perf.recall)):
                ax_pr_row[j].plot(perf.recall, perf.precision, "-", color="r")
                ax_pr_row[j].plot(opt.get("recall"), opt.get("precision"), "o", color="red")
                if opt_res_sec is not None:
                    ax_pr_row[j].plot(opt_sec.get("recall"), opt_sec.get("precision"), "o", color="black")
                ax_score_row[j].plot(perf[self.score_name], perf.precision, label="precision")
                ax_score_row[j].plot(perf[self.score_name], perf.recall, label="recall")
                ax_score_row[j].plot(perf[self.score_name], perf.f1, label="f1")

                title = cat
                ax_pr_row[j].set_title(title)
                ax_score_row[j].set_title(title)
                ax_pr_row[j].set_xlabel("Recall")
                ax_score_row[j].set_xlabel("score")
                ax_score_row[j].grid(True)
                ax_score_row[0].legend(loc="upper right")

                best_f1 = perf.f1.max()
                if not math.isnan(best_f1):
                    f1_range = self.__get_range_from_gap(best_f1)
                    ax_pr_row[j].set_xlim(f1_range)
                    ax_pr_row[j].set_ylim(f1_range)
                    max_score = max(perf[perf.f1 >= f1_range[0]][self.score_name].max(), 0.01)
                    ax_score_row[j].set_xlim([0, max_score])
                    ax_score_row[j].set_ylim(f1_range)
                ax_pr_row[j].grid(True)
        for i in range(num_empty_subplots):
            if ax_pr_row is not None:
                fig_pr.delaxes(ax_pr_row[m - i - 1])
                fig_score.delaxes(ax_score_row[m - i - 1])
        fig_pr.suptitle(f"Precision/Recall curve ({name})", fontsize=20)
        fig_score.suptitle(f"Score vs. accuracy ({name})", fontsize=20)
        fig_pr.tight_layout()
        fig_score.tight_layout()
        plt.show()

    @staticmethod
    def __get_range_from_gap(metric):
        gap = max(1 - metric, 0.01)
        min_ = max(metric - gap, 0)
        return min_, 1

    def get_performance(self, data: pd.DataFrame, categories: list[str]):
        perf_curve = {}
        opt_res = {}
        opt_tab = pd.DataFrame()
        error_types_table = pd.DataFrame()
        for cat in categories:
            d = self.__filter_by_category(data, cat)
            performance_dict, pr_curve = self.__calc_performance(d)
            perf_curve[cat] = pr_curve
            opt_res[cat] = performance_dict
            row = self.__get_general_performance_df(cat, performance_dict)
            opt_tab = pd.concat([opt_tab, row])
            if self.verbosity > 1:
                error_types_row = self.__get_error_types_row(cat, performance_dict)
                error_types_table = pd.concat([error_types_table, error_types_row])

        return opt_tab, opt_res, perf_curve, error_types_table

    @staticmethod
    def indel_analysis(
        data: pd.DataFrame,
        data_name: str,
        variable_names=("indel_length", "hmer_length", "max_vaf", "qual", "gq", "dp"),
        min_values=(1, 0, 0, 0, 0, 0),
        max_values=(15, 20, 1, 80, 80, 80),
        bin_widths=(1, 1, 0.05, 3, 3, 3),
        tick_widths=(2, 5, 0.1, 10, 10, 10),
    ):
        indels = data[data["indel"]].copy()
        if tick_widths is None:
            tick_widths = bin_widths
        hmer_indels_index = indels["hmer_length"] > 0
        hmer_indels = ("hmer_indels", indels[hmer_indels_index])
        non_hmer_indels = ("non_hmer_indels", indels[~hmer_indels_index])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k, variable_name in enumerate(variable_names):
                min_value = min_values[k]
                max_value = max_values[k]
                if max_value > 1:
                    max_value += 1
                bin_width = bin_widths[k]
                tick_width = tick_widths[k]

                if variable_name not in data.columns:
                    continue
                display(Markdown(f"### {variable_name}"))
                for indel_type, indels in (hmer_indels, non_hmer_indels):
                    display(Markdown(f"#### {data_name} {indel_type} - {variable_name}"))
                    _, ax = plt.subplots(1, 5, figsize=(15, 3))
                    classes = ["fp", "tp", "fn"]
                    bins = np.arange(min_value, max_value, bin_width)
                    ins_hist = {}
                    del_hist = {}
                    for i in range(3):
                        class_ = classes[i]
                        ax[i].set_title(class_)
                        vals = indels[indels[class_] & (indels["indel_classify"] == "ins")][variable_name]
                        ins_hist[class_], _, _ = ax[i].hist(
                            vals[~vals.isna()],
                            bins=bins,
                            alpha=0.5,
                            label="ins",
                        )
                        vals = indels[indels[class_] & (indels["indel_classify"] == "del")][variable_name]
                        del_hist[class_], _, _ = ax[i].hist(
                            vals[~vals.isna()],
                            bins=bins,
                            alpha=0.5,
                            color="g",
                            label="del",
                        )
                        ax[i].set_xlabel(variable_name)
                        ax[i].set_xlim(min_value, max_value)
                        ax[i].legend()
                        ax[i].xaxis.set_ticks(np.arange(min_value, max_value, tick_width))
                        if bin_width < 1:
                            ax[i].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    np.seterr(invalid="ignore")
                    precision_ins = ins_hist["tp"] / (ins_hist["tp"] + ins_hist["fp"])
                    precision_del = del_hist["tp"] / (del_hist["tp"] + del_hist["fp"])
                    recall_ins = ins_hist["tp"] / (ins_hist["tp"] + ins_hist["fn"])
                    recall_del = del_hist["tp"] / (del_hist["tp"] + del_hist["fn"])
                    ReportUtils.__plot_accuracy_metric(
                        ax[3],
                        bins,
                        min_value,
                        max_value,
                        precision_del,
                        precision_ins,
                        tick_width,
                        variable_name,
                        "precision",
                    )
                    ReportUtils.__plot_accuracy_metric(
                        ax[4], bins, min_value, max_value, recall_del, recall_ins, tick_width, variable_name, "recall"
                    )
                    plt.show()

    @staticmethod
    def make_multi_index(df):
        """We make the data-frames multi-index before saving them to h5, for backwards compatability"""
        df.columns = pd.MultiIndex.from_tuples([("whole genome", x) for x in df.columns])

    @staticmethod
    def get_anchor(anchor_id: str) -> str:
        return f"<a class ='anchor' id='{anchor_id}'> </a>"

    @staticmethod
    def __plot_accuracy_metric(fig, bins, min_value, max_value, metric_del, metric_ins, tick_width, x_label, title):
        fig.set_title(title)
        fig.plot(bins[1:], metric_ins, ".--", label="ins")
        fig.plot(bins[1:], metric_del, "g.--", label="del")
        fig.set_xlabel(x_label)
        fig.set_xlim(min_value, max_value)
        fig.set_ylim(0, 1)
        fig.legend()
        fig.xaxis.set_ticks(np.arange(min_value, max_value, tick_width))

    def __display_tables(self, opt_tab, error_types_tab, name, sec_error_types_tab=None):
        anchor = self.get_anchor(f"gen_acc_{name}")
        display(Markdown(f"### General accuracy ({name}) {anchor}"))
        if self.verbosity > 1:
            display(
                Markdown(
                    "* #pos - total variants in ground-truth\n"
                    "* neg - false-positive variants (before filtering)\n"
                    "* max_recall - fraction of true variants with correctly generated candidate"
                )
            )
        display(opt_tab)
        if self.verbosity > 1:
            anchor = self.get_anchor(f"err_types_{name}")
            display(Markdown(f"### Error types ({name}) {anchor}"))
            display(
                Markdown(
                    "* noise - called variants which have no matching true variant\n"
                    "* wrong_allele - called variants which have a true variant in "
                    "the same position with a non-matching allele\n"
                    "* hom->het - homozygous variants called as heterozygous\n"
                    "* het->hom - heterozygous variants called as homozygous\n"
                    "* miss - completely missed true variants\n"
                )
            )
            drop_list = [
                "hmer Indel 4",
                "hmer Indel 5",
                "hmer Indel 6",
                "hmer Indel 7",
                "hmer Indel 8",
                "non-hmer Indel w/o LCR",
            ]
            error_types_tab.drop(drop_list, errors="ignore", inplace=True)
            display(error_types_tab)
            if sec_error_types_tab is not None:
                display(Markdown("After systematic error correction:"))
                sec_error_types_tab.drop(drop_list, errors="ignore", inplace=True)
                display(sec_error_types_tab)
            error_types_tab.plot.bar(stacked=True, figsize=(8, 6))

    def __get_general_performance_df(self, cat, performance_dict):
        if self.verbosity > 1:
            return pd.DataFrame(
                {
                    "# pos": performance_dict["# pos"],
                    "# neg": performance_dict["initial_fp"],
                    "fn": performance_dict["initial_fn"],
                    "max recall": performance_dict["max_recall"],
                    "recall": performance_dict["recall"],
                    "precision": performance_dict["precision"],
                    "F1": performance_dict["f1"],
                },
                index=[cat],
            )
        return pd.DataFrame(
            {
                "true-vars": performance_dict["# pos"],
                "fn": performance_dict["initial_fn"],
                "fp": performance_dict["initial_fp"],
                "recall": performance_dict["recall"],
                "precision": performance_dict["precision"],
                "F1": performance_dict["f1"],
            },
            index=[cat],
        )

    @staticmethod
    def __get_error_types_row(cat, performance_dict):
        return pd.DataFrame(
            {
                "noise": performance_dict["noise"],
                "wrong_allele": performance_dict["wrong_allele"],
                "hom->het": performance_dict["hom->het"],
                "het->hom": performance_dict["het->hom"],
                "filter_true": performance_dict["filter_true"],
                "miss_candidate": performance_dict["miss_candidate"],
            },
            index=[cat],
        )

    @staticmethod
    def has_sec(x):
        res = False
        if x is not None and not pd.isna(x):
            if "SEC" in x:
                res = True
        return res

    def __calc_performance(self, data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        score_name = self.score_name
        d = data.copy()
        d = d[
            [
                "call",
                "base",
                score_name,
                "filter",
                "alleles",
                "gt_ultima",
                "gt_ground_truth",
                "tp",
                "fp",
                "fn",
                "error_type",
            ]
        ]
        d.loc[d["call"].isna(), "call"] = "NA"
        d.loc[d["base"].isna(), "base"] = "NA"
        # Change tree_score such that PASS will get high score values. FNs will be the lowest
        score_pass = d.query(f'filter == "PASS" & {score_name} == {score_name}').head(20)[score_name].mean()
        score_not_pass = d.query(f'filter != "PASS" & {score_name} == {score_name}').head(20)[score_name].mean()
        dir_switch = 1 if score_pass > score_not_pass else -1
        score = d[score_name] * dir_switch
        score = score - score.min()

        # define variants which didn't have any candidate
        missing_candidates_index = (d["base"] == "FN") & (d["call"] == "NA")
        missing_candidates = missing_candidates_index.sum()
        # Give missing candidates score of -1 (order them at the top for pr_curve)
        d[score_name] = np.where(missing_candidates_index, -1, score)

        # Calculate the precision and recall post filtering
        filtered_tp = len(d[d["tp"] & (d["filter"] != "PASS")])
        filtered_fp = len(d[d["fp"] & (d["filter"] != "PASS")])
        initial_fp = d["fp"].sum()
        initial_tp = d["tp"].sum()
        initial_fn = d["fn"].sum()
        total_variants = initial_tp + initial_fn
        fp = initial_fp - filtered_fp
        fn = initial_fn + filtered_tp
        tp = initial_tp - filtered_tp
        error_type = d["error_type"]
        noise = ((error_type == ErrorType.NOISE) & (d["filter"] == "PASS")).sum()
        hom_to_het = ((error_type == ErrorType.HOM_TO_HET) & (d["filter"] == "PASS")).sum()
        het_to_hom = ((error_type == ErrorType.HET_TO_HOM) & (d["filter"] == "PASS")).sum()
        wrong_allele = ((error_type == ErrorType.WRONG_ALLELE) & (d["filter"] == "PASS")).sum()
        filtered_true = fn - missing_candidates - hom_to_het - het_to_hom - wrong_allele

        recall = get_recall(fn, tp, np.nan)
        max_recall = get_recall(missing_candidates, (tp + fn - missing_candidates), np.nan)
        precision = get_precision(fp, tp, np.nan)
        f1 = get_f1(recall, precision, np.nan)

        pr_curve = pd.DataFrame()
        result_dict = {
            "# pos": total_variants,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "max_recall": max_recall,
            "initial_tp": initial_tp,
            "initial_fp": initial_fp,
            "initial_fn": initial_fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "noise": noise,
            "wrong_allele": wrong_allele,
            "hom->het": hom_to_het,
            "het->hom": het_to_hom,
            "filter_true": filtered_true,
            "miss_candidate": missing_candidates,
        }
        if len(d) < 10:
            return result_dict, pr_curve

        # Sort by score to get precision/recall curve
        d = d.sort_values(by=[score_name])

        # Calculate precision and recall continuously
        # how many true variants are either initial FNs (missing/filtered) or below score threshold
        d["fn"] = initial_fn + np.cumsum(d["tp"])
        d["tp"] = initial_tp - np.cumsum(d["tp"])  # how many tps pass score threshold
        d["fp"] = initial_fp - np.cumsum(d["fp"])  # how many fps pass score threshold
        d["recall"] = d[["fn", "tp"]].apply(lambda t: get_recall(t[0], t[1], np.nan), axis=1)
        d["precision"] = d[["fp", "tp"]].apply(lambda t: get_precision(t[0], t[1], np.nan), axis=1)
        d["f1"] = d[["precision", "recall"]].apply(lambda t: get_f1(t[0], t[1], np.nan), axis=1)
        pr_curve = d[[score_name, "recall", "precision", "f1"]]
        return result_dict, pr_curve

    @staticmethod
    def __filter_by_category(data, cat) -> pd.DataFrame:
        result = None
        if cat == "SNP":
            result = data[~data["indel"]]
        elif cat == "Indel":
            result = data[data["indel"]]
        elif cat == "non-hmer Indel":
            result = data[(data["indel"]) & (data["hmer_length"] == 0) & (data["indel_length"] > 0)]
        elif cat == "non-hmer Indel w/o LCR":
            result = data[(data["indel"]) & (data["hmer_length"] == 0) & (data["indel_length"] > 0) & (~data["LCR"])]
        elif cat == "hmer Indel <=4":
            result = data[(data["indel"]) & (data["hmer_length"] > 0) & (data["hmer_length"] <= 4)]
        elif cat == "hmer Indel >4,<=8":
            result = data[(data["indel"]) & (data["hmer_length"] > 4) & (data["hmer_length"] <= 8)]
        elif cat == "hmer Indel >8,<=10":
            result = data[(data["indel"]) & (data["hmer_length"] > 8) & (data["hmer_length"] <= 10)]
        elif cat == "hmer Indel >10,<=12":
            result = data[(data["indel"]) & (data["hmer_length"] > 10) & (data["hmer_length"] <= 12)]
        elif cat == "hmer Indel >12,<=14":
            result = data[(data["indel"]) & (data["hmer_length"] > 12) & (data["hmer_length"] <= 14)]
        elif cat == "hmer Indel >15,<=19":
            result = data[(data["indel"]) & (data["hmer_length"] > 14) & (data["hmer_length"] <= 19)]
        elif cat == "hmer Indel >=20":
            result = data[(data["indel"]) & (data["hmer_length"] >= 20)]
        for i in range(1, 10):
            if cat == "hmer Indel {0:d}".format(i):
                result = data[(data["indel"]) & (data["hmer_length"] == i)]
                return result
        if result is None:
            raise RuntimeError(f"No such category: {cat}")
        return result
