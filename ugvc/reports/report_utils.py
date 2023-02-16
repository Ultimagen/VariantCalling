from __future__ import annotations

import math
from configparser import ConfigParser
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from ugvc.utils.stats_utils import get_f1, get_precision, get_recall


def parse_config(config_file):
    parser = ConfigParser()
    parser.read(config_file)
    param_names = ["run_id", "pipeline_version", "h5_concordance_file"]
    parameters = {p: parser.get("VarReport", p) for p in param_names}
    parameters["verbosity"] = parser.get("VarReport", "verbosity", fallback="1")
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
    MISS = 2
    HOM_TO_HET = 3
    HET_TO_HOM = 4
    WRONG_ALLELE = 5
    NO_ERROR = 6


class ShortReportUtils:
    def __init__(self, verbosity, h5outfile: str, num_plots_in_row=5, min_value=0.2):
        self.verbosity = verbosity
        self.h5outfile = h5outfile
        self.min_value = min_value
        self.num_plots_in_row = num_plots_in_row
        self.score_name = "tree_score"

    def basic_analysis(self, data: pd.DataFrame, categories, out_key, out_key_sec=None) -> None:
        data_sec = None
        if out_key_sec is not None:
            sec_df = data.copy()
            if "blacklst" in sec_df.columns:
                is_sec = sec_df["blacklst"].apply(self.has_sec)
                sec_df.loc[is_sec, "filter"] = "SEC"
                sec_df.loc[is_sec & (sec_df["classify_gt"] == "tp"), "classify_gt"] = "fn"
                data_sec = sec_df[~(is_sec & (sec_df["classify_gt"] == "fp"))]

        opt_tab, opt_res, perf_curve, detailed_fp_tab = self.get_performance(data, categories)
        opt_tab.to_hdf(self.h5outfile, key=out_key)
        detailed_fp_tab.to_hdf(self.h5outfile, key=f"{out_key}_fp_details")

        if data_sec is not None:
            sec_opt_tab, sec_opt_res, _, detailed_fp_tab = self.get_performance(data_sec, categories)
            if self.verbosity > 1:
                self.plot_performance(
                    perf_curve,
                    opt_res,
                    categories,
                    opt_res_sec=sec_opt_res,
                )
            sec_opt_tab.to_hdf(self.h5outfile, key=out_key_sec)
            detailed_fp_tab.to_hdf(self.h5outfile, key=f"{out_key_sec}_fp_details")
            display(sec_opt_tab)
            display(detailed_fp_tab)
        else:
            self.plot_performance(perf_curve, opt_res, categories)
            display(opt_tab)
            display(detailed_fp_tab)

    def homozygous_genotyping_analysis(self, d: pd.DataFrame, categories: list[str], out_key: str) -> None:
        hmz_data = d[(d["gt_ground_truth"] == (1, 1)) & (d["classify"] != "fn")]
        opt_tab, _, _, _ = self.get_performance(hmz_data, categories)
        opt_tab.to_hdf(self.h5outfile, out_key)
        display(opt_tab)

    def base_stratification_analysis(self, d: pd.DataFrame, categories: list[str], bases: tuple) -> pd.DataFrame:
        base_data = d[
            (~d["indel"] & ((d["ref"] == bases[0]) | (d["ref"] == bases[1])))
            | ((d["hmer_indel_length"] > 0) & ((d["hmer_indel_nuc"] == bases[0]) | (d["hmer_indel_nuc"] == bases[1])))
        ]

        opt_tab, opt_res, perf_curve, _ = self.get_performance(base_data, categories)
        opt_tab.rename(index={a: "{0} ({1}/{2})".format(a, bases[0], bases[1]) for a in opt_tab.index}, inplace=True)
        if self.verbosity > 1:
            self.plot_performance(perf_curve, opt_res, categories)

        display(opt_tab)
        return opt_tab

    def plot_performance(
        self,
        perf_curve: dict,
        opt_res: dict,
        categories: list[str],
        opt_res_sec=None,
    ):
        m = self.num_plots_in_row
        n = math.ceil(len(categories) / m)
        num_empty_subplots = n * m - len(categories)

        fig, ax = plt.subplots(n, m, figsize=(3 * m, 3 * n + 0.5 * (n - 1)))
        opt_sec = None
        ax_row = None
        for cat_index, cat in enumerate(categories):
            i = math.floor(cat_index / m)
            j = cat_index % m
            if len(ax.shape) == 2:
                ax_row = ax[i]
            else:
                ax_row = ax
            ax_row[0].set_ylabel("Precision")
            perf = perf_curve[cat]
            opt = opt_res[cat]
            if opt_res_sec is not None:
                opt_sec = opt_res_sec[cat]
            if not perf.empty:
                ax_row[j].plot(perf.recall, perf.precision, "-", color="r")
                ax_row[j].plot(opt.get("recall"), opt.get("precision"), "o", color="red")
                if opt_res_sec is not None:
                    ax_row[j].plot(opt_sec.get("recall"), opt_sec.get("precision"), "o", color="black")

            title = cat
            ax_row[j].set_title(title)
            ax_row[j].set_xlabel("Recall")
            ax_row[j].set_xlim([self.min_value, 1])
            ax_row[j].set_ylim([self.min_value, 1])
            ax_row[j].grid(True)
        for i in range(num_empty_subplots):
            if ax_row is not None:
                fig.delaxes(ax_row[m - i - 1])

        plt.tight_layout()
        plt.show()

    def get_performance(self, data: pd.DataFrame, categories: list[str]):
        perf_curve = {}
        opt_res = {}
        opt_tab = pd.DataFrame()
        detailed_fp_table = pd.DataFrame()
        for cat in categories:
            d = self.__filter_by_category(data, cat)
            performance_dict, pr_curve = self.__calc_performance(d)
            perf_curve[cat] = pr_curve
            opt_res[cat] = performance_dict
            row = self.__get_general_performance_df(cat, performance_dict)
            opt_tab = pd.concat([opt_tab, row])
            if self.verbosity > 1:
                detailed_fp_row = self.__get_detailed_fp_df(cat, performance_dict)
                detailed_fp_table = pd.concat([detailed_fp_table, detailed_fp_row])

        return opt_tab, opt_res, perf_curve, detailed_fp_table

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

    def __get_detailed_fp_df(self, cat, performance_dict):
        return pd.DataFrame(
            {
                "fp": performance_dict["initial_fp"],
                "o->e": performance_dict["hom->het"],
                "e->o": performance_dict["het->hom"],
                "ale_err": performance_dict["wrong_allele"],
                "fp(filter)": performance_dict["fp"],
                "precision": performance_dict["precision"],
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
        score_name = "tree_score"
        d = data.copy()
        d = d[["call", "base", score_name, "filter", "alleles", "gt_ultima", "gt_ground_truth"]]
        d.loc[d["call"].isna(), "call"] = "NA"
        d.loc[d["base"].isna(), "base"] = "NA"
        # Change tree_score such that PASS will get high score values. FNs will be the lowest
        score_pass = d.query(f'filter == "PASS" & {score_name} == {score_name}').head(20)[score_name].mean()
        score_not_pass = d.query(f'filter != "PASS" & {score_name} == {score_name}').head(20)[score_name].mean()
        dir_switch = 1 if score_pass > score_not_pass else -1
        score = d[score_name] * dir_switch
        score = score - score.min()

        d["fp"] = (d["call"] == "FP") | (d["call"] == "FP_CA")
        d["fn"] = (d["base"] == "FN") | (d["base"] == "FN_CA")
        d["tp"] = d["call"] == "TP"

        # define variants which didn't have any candidate
        missing_candidates_index = (d["base"] == "FN") & (d["call"] == "NA")
        missing_candidates = missing_candidates_index.sum()
        # Give missing candidates score of -1 (order them at the top for pr_curve)
        d[self.score_name] = np.where(missing_candidates_index, -1, score)

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

        genotypes = d["gt_ground_truth"] + d["gt_ultima"]
        error_type = genotypes.apply(self.get_error_type)
        hom_to_het = (error_type == ErrorType.HOM_TO_HET).sum()
        het_to_hom = (error_type == ErrorType.HET_TO_HOM).sum()
        wrong_allele = (error_type == ErrorType.WRONG_ALLELE).sum()

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
            "hom->het": hom_to_het,
            "het->hom": het_to_hom,
            "wrong_allele": wrong_allele,
        }
        if len(d) < 10:
            return result_dict, pr_curve

        # Sort by score to get precision/recall curve
        d = d.sort_values(by=[self.score_name])

        # Calculate precision and recall continuously
        # how many true variants are either initial FNs (missing/filtered) or below score threshold
        d["fn"] = initial_fn + np.cumsum(d["tp"])
        d["tp"] = tp - np.cumsum(d["tp"])  # how many tps pass score threshold
        d["fp"] = fp - np.cumsum(d["fp"])  # how many fps pass score threshold
        d["recall"] = d["tp"] / (d["fn"] + d["tp"])
        d["precision"] = d["tp"] / (d["fp"] + d["tp"])

        # Select for pr_curve variants which are not missed candidates / first 20 pos/neg variants
        d["mask"] = ((d["tp"] + d["fn"]) >= 20) & ((d["tp"] + d["fp"]) >= 20) & (d[self.score_name] >= 0)
        if len(d[d["mask"]]) > 0:
            pr_curve = d[["fp", "fn", "tp", "recall", "precision"]][d["mask"]]
        return result_dict, pr_curve

    @staticmethod
    def __filter_by_category(data, cat) -> pd.DataFrame:
        result = None
        if cat == "SNP":
            result = data[~data["indel"]]
        elif cat == "Indel":
            result = data[data["indel"]]
        elif cat == "non-hmer Indel":
            result = data[(data["indel"]) & (data["hmer_indel_length"] == 0) & (data["indel_length"] > 0)]
        elif cat == "non-hmer Indel w/o LCR":
            result = data[
                (data["indel"]) & (data["hmer_indel_length"] == 0) & (data["indel_length"] > 0) & (~data["LCR"])
            ]
        elif cat == "hmer Indel <=4":
            result = data[(data["indel"]) & (data["hmer_indel_length"] > 0) & (data["hmer_indel_length"] <= 4)]
        elif cat == "hmer Indel >4,<=8":
            result = data[(data["indel"]) & (data["hmer_indel_length"] > 4) & (data["hmer_indel_length"] <= 8)]
        elif cat == "hmer Indel >8,<=10":
            result = data[(data["indel"]) & (data["hmer_indel_length"] > 8) & (data["hmer_indel_length"] <= 10)]
        elif cat == "hmer Indel >10,<=14":
            result = data[(data["indel"]) & (data["hmer_indel_length"] > 10) & (data["hmer_indel_length"] <= 14)]
        elif cat == "hmer Indel >15,<=19":
            result = data[(data["indel"]) & (data["hmer_indel_length"] > 14) & (data["hmer_indel_length"] <= 19)]
        elif cat == "hmer Indel >=20":
            result = data[(data["indel"]) & (data["hmer_indel_length"] >= 20)]
        for i in range(1, 10):
            if cat == "hmer Indel {0:d}".format(i):
                result = data[(data["indel"]) & (data["hmer_indel_length"] == i)]
                return result
        if result is None:
            raise RuntimeError(f"No such category: {cat}")
        return result

    @staticmethod
    def get_error_type(genotype_pair: tuple) -> ErrorType:
        gtr_gt = set(genotype_pair[0:2])
        call_gt = set(genotype_pair[2:4])

        if gtr_gt == call_gt:
            return ErrorType.NO_ERROR

        if gtr_gt in ({0}, {None}):
            return ErrorType.NOISE

        if call_gt in ({0}, {None}):
            return ErrorType.MISS

        if gtr_gt.intersection(call_gt) == gtr_gt:
            return ErrorType.HOM_TO_HET

        if gtr_gt.intersection(call_gt) == call_gt:
            return ErrorType.HET_TO_HOM

        return ErrorType.WRONG_ALLELE
