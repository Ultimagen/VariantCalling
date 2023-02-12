from configparser import ConfigParser

import matplotlib.pyplot as plt
import nexusplt as nxp
import numpy as np
import pandas as pd


def parse_config(config_file):
    parser = ConfigParser()
    parser.read(config_file)
    param_names = [
        "run_id",
        "pipeline_version",
        "h5_concordance_file",
    ]
    parameters = {p: parser.get("VarReport", p) for p in param_names}

    # Optional parameters
    parameters["reference_version"] = parser.get("VarReport", "reference_version", fallback="hg38")
    parameters["truth_sample_name"] = parser.get("VarReport", "truth_sample_name", fallback="NA")
    parameters["image_prefix"] = (
        parser.get("VarReport", "image_output_prefix", fallback=parameters["run_id"] + ".vars") + "."
    )
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


class ShortReportUtils:
    def __init__(self, image_dir, image_prefix):

        self.image_dir = image_dir
        self.image_prefix = image_prefix

    def plot_performance(
        self, perf_curve, opt_res, categories, sources: dict, ext=None, img=None, legend=None, opt_res_sec=None
    ):
        n = len(categories)
        fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
        col = ["r", "b", "g", "m", "k"]

        for i, cat in enumerate(categories):
            for j, s in enumerate(sources):
                perf = perf_curve[s][cat]
                opt = opt_res[s][cat]
                if opt_res_sec is not None:
                    opt_sec = opt_res_sec[s][cat]
                if not perf.empty:
                    ax[i].plot(perf.recall, perf.precision, "-", label=s, color=col[j])
                    ax[i].plot(opt.get("recall"), opt.get("precision"), "o", color=col[j])
                    if opt_res_sec is not None:
                        ax[i].plot(opt_sec.get("recall"), opt_sec.get("precision"), "o", color="black")
                title = cat if ext is None else "{0} ({1})".format(cat, ext)
                ax[i].set_title(title)
                ax[i].set_xlabel("Recall")
                ax[i].set_xlim([0.6, 1])
                ax[i].set_ylim([0.6, 1])
                ax[i].grid(True)

        ax[0].set_ylabel("Precision")
        if legend:
            ax[0].legend(loc="lower left")
        if img:
            nxp.save(fig, self.image_prefix + img, "png", outdir=self.image_dir)

    def get_performance(self, data, categories, sources):
        opt_tab = {}
        opt_res = {}
        perf_curve = {}
        for s in sources:
            opt_tab[s] = pd.DataFrame()
            opt_res[s] = {}
            perf_curve[s] = {}

            for cat in categories:
                d = self.__filter_by_category(data[s], cat)
                performance_dict = self.__calc_performance(d)
                pr_curve = performance_dict["pr_curve"]
                perf_curve[s][cat] = pr_curve
                opt_res[s][cat] = performance_dict
                # Report table columns
                row = pd.DataFrame(
                    {
                        "# pos": performance_dict["initial_tp"] + performance_dict["initial_fn"],
                        "# neg": performance_dict["initial_fp"],
                        "# init fn": performance_dict["initial_fn"],
                        "max recall": performance_dict["max_recall"],
                        "recall": performance_dict["recall"],
                        "precision": performance_dict["precision"],
                        "F1": performance_dict["f1"],
                    },
                    index=[cat],
                )
                opt_tab[s] = pd.concat([opt_tab[s], row])

        return opt_tab, opt_res, perf_curve

    @staticmethod
    def has_sec(x):
        res = False
        if x is not None and not pd.isna(x):
            if "SEC" in x:
                res = True
        return res

    @staticmethod
    def __calc_performance(data: pd.DataFrame) -> dict:
        score_name = "tree_score"
        d = data.copy()
        d = d[["call", "base", score_name, "filter"]]
        d.loc[d["call"].isna(), "call"] = "NA"
        d.loc[d["base"].isna(), "base"] = "NA"
        # Change tree_score such that PASS will get high score values. FNs will be the lowest
        # score_pass = d[(d['filter']=="PASS") & (d[score_name].isna())].head(20)[score_name].mean()
        # score_not_pass =  d[(d['filter']!="PASS") & (d[score_name].isna())].head(20)[score_name].mean()
        score_pass = d.query('filter=="PASS" & tree_score==tree_score').head(20)["tree_score"].mean()
        score_not_pass = d.query('filter!="PASS" & tree_score==tree_score').head(20)["tree_score"].mean()
        dir_switch = 1 if score_pass > score_not_pass else -1
        score = d[score_name] * dir_switch
        score = score - score.min()

        d["fp"] = (d["call"] == "FP") | (d["call"] == "FP_CA")
        d["fn"] = (d["base"] == "FN") | (d["base"] == "FN_CA")
        d["tp"] = (d["base"] == "TP") & (d["call"] == "TP")

        # define variants which didn't have any candidate
        missing_candiddates_index = (d["base"] == "FN") & (d["call"] == "NA")
        missing_candidates = missing_candiddates_index.sum()
        # Give missing candidates score of -1 (order them at the top for pr_curve)
        d[score_name] = np.where(missing_candiddates_index, -1, score)

        # Calculate the precision and recall as ouputted by the model (based on the FILTER column)
        filtered_tp = len(d[d["tp"] & (d["filter"] != "PASS")])
        filtered_fp = len(d[d["fp"] & (d["filter"] != "PASS")])
        initial_fp = d["fp"].sum()
        initial_tp = d["tp"].sum()
        initial_fn = d["fn"].sum()
        fp = initial_fp - filtered_fp
        fn = initial_fn + filtered_tp
        tp = initial_tp - filtered_tp

        recall = tp / (tp + fn) if (tp + fn > 0) else np.nan
        precision = tp / (tp + fp) if (tp + fp > 0) else np.nan
        f1 = tp / (tp + 0.5 * fn + 0.5 * fp)
        max_recall = (tp + fn - missing_candidates) / (tp + fn)
        result_dict = {
            "pr_curve": pd.DataFrame(),
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
        }
        if len(d) < 10:
            return result_dict

        # Sort by score to get precision/recall curve
        d = d.sort_values(by=[score_name])

        # Calculate precision and recall continuously

        # how many true variants are either initial FNs (missing/filtered) or below score threshold
        d["fn"] = initial_fn + np.cumsum(d["tp"])
        d["tp"] = tp - np.cumsum(d["tp"])  # how many tps pass score threshold
        d["fp"] = fp - np.cumsum(d["fp"])  # how many fps pass score threshold
        d["recall"] = d["tp"] / (d["fn"] + d["tp"])
        d["precision"] = d["tp"] / (d["fp"] + d["tp"])

        # Select for pr_curve variants which are not missed candidates / first 20 pos/neg variants
        d["mask"] = ((d["tp"] + d["fn"]) >= 20) & ((d["tp"] + d["fp"]) >= 20) & (d[score_name] >= 0)
        if len(d[d["mask"]]) > 0:
            result_dict["pr_curve"] = d[["fp", "fn", "tp", "recall", "precision"]][d["mask"]]
        return result_dict

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
