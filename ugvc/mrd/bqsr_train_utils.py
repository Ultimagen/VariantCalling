from __future__ import annotations

import json
import os
import subprocess
from enum import Enum
from os.path import basename, isfile
from os.path import join as pjoin

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
import xgboost as xgb
from simppl.simple_pipeline import SimplePipeline
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ugvc import logger
from ugvc.dna.format import CHROM_DTYPE, DEFAULT_FLOW_ORDER
from ugvc.dna.utils import get_max_softclip_len, revcomp
from ugvc.srsnv.srsnv_utils import balance_df_according_motif, plot_precision_recall, precision_recall_curve
from ugvc.utils.consts import FileExtension
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json
from ugvc.vcfbed.variant_annotation import get_cycle_skip_dataframe, get_motif_around

default_model_params = {
    "balance_motifs_in_train_set": True,
    "sample_size": 50000000,
    "forest_size": 100,
    "eta": 0.1,
    "max_depth": 10,
    "n_jobs": 16,
    "subsample": 0.35,
    "motif_only_features": ["ref_alt_motif", "left_motif", "right_motif"],
    "categorical_features": ["cycle_skip_status"],
    "non_categorical_features": [
        "X_SCORE",
        "X_EDIST",
        "X_LENGTH",
        "X_MAPQ",
        "X_INDEX",
        "max_softclip_len",
        "X_FC1",
    ],
    "train_size": 100000,
    "test_size": 10000,
}


class FeatureMapFields(Enum):
    READ_COUNT = "X_READ_COUNT"
    FILTERED_COUNT = "X_FILTERED_COUNT"


def exec_command_list(commands: list[str], simple_pipeline: SimplePipeline = None):
    """
    Execute a list of commands

    Parameters
    ----------
    commands : list[str]
        List of commands to execute
    simple_pipeline : SimplePipeline, optional
        SimplePipeline object to use for printing and running commands, by default None
    """
    if simple_pipeline:
        for command in commands:
            simple_pipeline.print_and_run(command)
    else:
        for command in commands:
            logger.debug(command)
            subprocess.call(command, shell=True)


def prepare_featuremap_for_training(
    workdir: str,
    input_featuremap_vcf: str,
    training_set_size: int,
    sp: SimplePipeline = None,
    regions_file: str = None,
    cram_stats_file: str = None,
    read_effective_coverage_from_sorter_json_kwargs: dict = None,
):
    """
    Prepare a featuremap for training,

    Parameters
    ----------
    workdir : str
        Working directory to which outputs will be written
    input_featuremap_vcf : str
        Path to input featuremap
    training_set_size : int
        Size of training set to create, must be at larger than 0
    sp : SimplePipeline, optional
        SimplePipeline object to use for printing and running commands
    regions_file : str, optional
        Path to regions file, by default None
    cram_stats_file : str, optional
        Path to cram stats file, by default None
    read_effective_coverage_from_sorter_json_kwargs : dict, optional
        kwargs for read_effective_coverage_from_sorter_json, by default None

    Returns
    -------
    str
        Path to sampled featuremap

    """

    # check inputs define paths
    os.makedirs(workdir, exist_ok=True)
    assert isfile(input_featuremap_vcf), f"input_featuremap_vcf {input_featuremap_vcf} not found"
    assert input_featuremap_vcf.endswith(FileExtension.VCF_GZ.value)
    assert training_set_size > 0, f"training_set_size must be > 0, got {training_set_size}"
    if not read_effective_coverage_from_sorter_json_kwargs:
        read_effective_coverage_from_sorter_json_kwargs = {}
    # make sure X_READ_COUNT is in the INFO fields in the header
    with pysam.VariantFile(input_featuremap_vcf) as fmap:
        assert FeatureMapFields.READ_COUNT.value in fmap.header.info
    intersect_featuremap_vcf = pjoin(
        workdir,
        basename(input_featuremap_vcf).replace(FileExtension.VCF_GZ.value, ".intersect.vcf.gz"),
    )
    downsampled_featuremap_vcf = pjoin(
        workdir,
        basename(intersect_featuremap_vcf).replace(FileExtension.VCF_GZ.value, ".downsampled.vcf.gz"),
    )
    assert not (isfile(intersect_featuremap_vcf)), f"intersect_featuremap_vcf {intersect_featuremap_vcf} already exists"
    assert not (
        isfile(downsampled_featuremap_vcf)
    ), f"sample_featuremap_vcf_sorted {downsampled_featuremap_vcf} already exists"

    # filter featuremap - intersect with bed file and require coverage in range
    commandstr = f"bcftools view {input_featuremap_vcf}"
    if cram_stats_file:
        # get min and max coverage from cram stats file
        (
            _,
            _,
            _,
            min_coverage_for_fp,
            max_coverage_for_fp,
        ) = read_effective_coverage_from_sorter_json(cram_stats_file, **read_effective_coverage_from_sorter_json_kwargs)
        # filter out variants with coverage outside of min and max coverage
        commandstr = (
            commandstr
            + f" -i' (INFO/{FeatureMapFields.READ_COUNT.value} >= {min_coverage_for_fp}) && (INFO/{FeatureMapFields.READ_COUNT.value} <= {max_coverage_for_fp}) '"
        )
    if regions_file:
        commandstr = commandstr + f" -R {regions_file} "
    commandstr = commandstr + f" -O z -o {intersect_featuremap_vcf}"
    commandslist = [commandstr, f"bcftools index -t {intersect_featuremap_vcf}"]
    exec_command_list(commandslist, sp)
    assert os.path.isfile(intersect_featuremap_vcf), f"failed to create {intersect_featuremap_vcf}"

    # count entries in intersected featuremap and determine downsampling rate
    with pysam.VariantFile(intersect_featuremap_vcf) as fmap:
        featuremap_entry_number = 0
        for _ in enumerate(fmap.fetch()):
            featuremap_entry_number += 1
    downsampling_rate = training_set_size / featuremap_entry_number
    logger.info(
        f"training_set_size={training_set_size}, featuremap_entry_number = {featuremap_entry_number}, downsampling_rate {downsampling_rate}"
    )

    # random sample of intersected featuremap - sampled featuremap
    np.random.seed(0)
    sampling_array = np.random.uniform(size=featuremap_entry_number) < downsampling_rate
    print(sampling_array)
    with pysam.VariantFile(input_featuremap_vcf) as vcf_in:
        # Add a comment to the output file indicating the subsampling
        header = vcf_in.header
        header.add_line(
            f"##target_size=featuremap_entry_number, subsampled approximately {training_set_size} variants from a total of {featuremap_entry_number} at a rate of {downsampling_rate}"
        )
        with pysam.VariantFile(downsampled_featuremap_vcf, "w", header=header) as vcf_out:
            for j, rec in enumerate(vcf_in.fetch()):
                print(sampling_array[j])
                if sampling_array[j]:
                    vcf_out.write(rec)
    pysam.tabix_index(downsampled_featuremap_vcf, preset="vcf", force=True)
    assert isfile(downsampled_featuremap_vcf), f"failed to create {downsampled_featuremap_vcf}"
    assert isfile(downsampled_featuremap_vcf + ".tbi"), f"failed to create {downsampled_featuremap_vcf}.tbi"

    return downsampled_featuremap_vcf, featuremap_entry_number


class BQSRTrain:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        hom_snv_featuremap,
        single_substitution_featuremap,
        hom_snv_regions,
        single_sub_regions,
        cram_stats_file,
        out_path,
        out_basename,
        reference_fasta,
        flow_order,
        model_params_file,
        simple_pipeline=None,
    ):
        os.makedirs(out_path, exist_ok=True)
        if len(out_basename) > 0 and not out_basename.endswith("."):
            out_basename += "."

        self.hom_snv_featuremap = hom_snv_featuremap
        self.single_substitution_featuremap = single_substitution_featuremap
        self.hom_snv_regions = hom_snv_regions
        self.single_sub_regions = single_sub_regions
        self.cram_stats_file = cram_stats_file
        self.workdir = out_path
        self.out_basename = out_basename
        self.reference_fasta = reference_fasta
        self.flow_order = flow_order
        if model_params_file:
            with open(model_params_file, encoding="utf-8") as f:
                self.model_params = json.load(f)
        else:
            self.model_params = default_model_params
        self.data_name = single_substitution_featuremap.split("/")[-1].split(".featuremap")[0]

        self.sp = simple_pipeline

        self.model_save_path = f"{self.workdir}/{self.out_basename}model.joblib"
        self.X_test_save_path = f"{self.workdir}/{self.out_basename}X_test.parquet"
        self.y_test_save_path = f"{self.workdir}/{self.out_basename}y_test.parquet"
        self.X_train_save_path = f"{self.workdir}/{self.out_basename}X_train.parquet"
        self.y_train_save_path = f"{self.workdir}/{self.out_basename}y_train.parquet"
        self.params_save_path = f"{self.workdir}/{self.out_basename}params.json"

        self.f_fp_training = None
        self.f_tp_training = None

        if self.model_params["test_size"] < 10000:
            logger.error("test_size must be >= 10000")
            raise ValueError
        if self.model_params["train_size"] < 100000:
            logger.error("train_size must be >= 100000")
            raise ValueError

    def balance_motifs(self, data, plotme=False):
        tp_df = data[data["label"] == "TP"]
        fp_df = data[data["label"] == "FP"]

        balanced_tps = balance_df_according_motif(tp_df, "ref_alt_motif", 0)
        if plotme:
            plot_motif_balance(tp_df, "tps before balancing")
            plot_motif_balance(balanced_tps, "tps after balacing")
            plot_motif_balance(fp_df, "fps")
        data = pd.concat([balanced_tps, fp_df], ignore_index=True)

        return data

    def extract_data_from_featuremap_local(self):
        f_out = pjoin(self.workdir, f"{self.data_name}.training_data.parquet")
        if not isfile(f_out):
            # load fp-tp datasets to pd dataframes
            featuremap_info_fields = {
                "X_CIGAR": str,
                "X_EDIST": int,
                "X_FC1": int,
                "X_FC2": str,
                "X_READ_COUNT": int,
                "X_FILTERED_COUNT": int,
                "X_FLAGS": int,
                "X_LENGTH": int,
                "X_MAPQ": float,
                "X_INDEX": int,
                "X_RN": str,
                "X_SCORE": float,
                "rq": float,
                "as": str,
                "ts": str,
                "ae": str,
                "te": str,
                "s2": str,
                "a3": str
                # "ML_QUAL": float
            }

            logger.info(f"load {self.f_fp_training}")
            fp_parquet = pjoin(self.workdir, "fp_featuremap_to_dataframe.parquet")
            if not isfile(fp_parquet):
                df_fp = featuremap_to_dataframe_mrd_analysis_changes(
                    featuremap_vcf=self.f_fp_training,
                    output_file=fp_parquet,
                    reference_fasta=self.reference_fasta,
                    flow_order=self.flow_order,
                    info_fields_override=featuremap_info_fields,
                )
            else:
                df_fp = pd.read_parquet(fp_parquet)

            logger.info(f"load {self.f_tp_training}")
            tp_parquet = pjoin(self.workdir, "tp_featuremap_to_dataframe.parquet")
            if not isfile(tp_parquet):
                df_tp = featuremap_to_dataframe_mrd_analysis_changes(
                    featuremap_vcf=self.f_tp_training,
                    output_file=tp_parquet,
                    reference_fasta=self.reference_fasta,
                    flow_order=self.flow_order,
                    info_fields_override=featuremap_info_fields,
                )
            else:
                df_tp = pd.read_parquet(tp_parquet)

            # concat the tp and fp dataframes
            df = pd.concat((df_fp.assign(label="FP"), df_tp.assign(label="TP")))
            df = df.reset_index().set_index(["chrom", "pos"]).sort_index()
            df.drop("chrY", level=0, axis=0, inplace=True)
            df.drop("chrX", level=0, axis=0, inplace=True)
            df.replace(to_replace="None", value=np.nan, inplace=True)

            # add mixed tags
            balanced_epcr_cols = ["a_start", "ts", "ae", "te", "s2", "a3"]
            df = (
                df.rename(columns={"as": "a_start"})
                .fillna({x: 0 for x in balanced_epcr_cols})
                .astype({x: int for x in balanced_epcr_cols})
                .astype({"rq": float})
            )
            df = df.assign(
                is_reverse=(df["X_FLAGS"] & 16).astype(bool),
                qc_valid=~(df["X_FLAGS"] & 512).astype(bool),
                read_end_reached=~df["s2"].isnull(),
                strand_ratio_5=df["ts"] / (df["ts"] + df["a_start"]),
                strand_ratio_3=df["te"] / (df["te"] + df["ae"]),
            )

            is_mixed_query = (
                "a_start >= 2 and a_start <= 4 and ts >= 2 and ts <= 4 and ae >= 2 and ae <= 4 and te >= 2 and te <= 4"
            )
            df = df.assign(is_mixed=df.eval(is_mixed_query).values)

            # cycle skip tag
            df = df.assign(is_cycle_skip=df["cycle_skip_status"] == "cycle-skip")

            df.to_parquet(f_out)
            data = df
        else:
            logger.info(f"loading file {f_out}\n ...")
            data = pd.read_parquet(f_out)

        data.loc[:, "ref_alt_bases"] = data["ref"].str.cat(data["alt"], sep="")
        data.loc[:, "ref_alt_motif"] = data["ref_motif"].str.cat(data["alt"], sep="")
        data.loc[:, "left_motif"] = data["left_motif"].str.slice(start=0, stop=3)
        data.loc[:, "right_motif"] = data["right_motif"].str.slice(start=1, stop=4)
        data.loc[:, "X_FC1"] = data["X_FC1"].apply(int)

        return data

    def train_test_split_head_tail(self, data, label_columns):
        df_tp = data.query("label==1")
        df_fp = data.query("label==0")

        n_train = int(self.model_params["train_size"] / 2)
        n_test = int(self.model_params["test_size"] / 2)
        df_train = pd.concat((df_tp.head(n_train), df_fp.head(n_train)))
        df_test = pd.concat((df_tp.tail(n_test), df_fp.tail(n_test)))

        return df_train, df_test, df_train[label_columns], df_test[label_columns]

    def save_model_and_data(self, xgb_classifier, X_test, y_test, X_train, y_train):
        joblib.dump(xgb_classifier, self.model_save_path)

        for d, savename in (
            (X_test, self.X_test_save_path),
            (y_test, self.y_test_save_path),
            (X_train, self.X_train_save_path),
            (y_train, self.y_train_save_path),
        ):
            d.to_parquet(savename)

        params_for_save = {
            item[0]: item[1] for item in self.__dict__.items() if not isinstance(item[1], SimplePipeline)
        }
        with open(self.params_save_path, "w", encoding="utf-8") as f:
            json.dump(params_for_save, f)

    def process(self):
        # prepare featuremaps for training
        featuremaplist = [self.single_substitution_featuremap, self.hom_snv_featuremap]
        regionslist = [self.single_sub_regions, self.hom_snv_regions]
        statlist = [self.cram_stats_file, None]

        # training set of size will include n TP and n FP where n ~ 0.5*(test_size+train_size)
        desired_n = int(0.51 * (self.model_params["test_size"] + self.model_params["train_size"]))

        featuremaps_for_training = []
        total_n_list = []
        for featuremap, regions_file, cram_stats_file in zip(featuremaplist, regionslist, statlist):
            featuremap_for_training, total_n = prepare_featuremap_for_training(
                self.sp,
                self.workdir,
                featuremap,
                desired_n,
                regions_file,
                cram_stats_file,
                featuremap_entry_number=None,
            )
            if not (isfile(featuremap_for_training)):
                logger.error(f"failed to create training data from {featuremap}")
                raise ValueError
            featuremaps_for_training.append(featuremap_for_training)
            total_n_list.append(total_n)

        self.f_fp_training = featuremaps_for_training[0]
        self.f_tp_training = featuremaps_for_training[1]

        self.total_n_single_sub_featuremap = total_n_list[0]

        # load training data
        data = self.extract_data_from_featuremap_local()
        logger.info(f"Labels in data: \n{data['label'].value_counts()}")

        # xgb model: init and train
        motif_only_features = self.model_params["motif_only_features"]
        categorical_features = self.model_params["categorical_features"] + motif_only_features
        non_categorical_features = self.model_params["non_categorical_features"]

        features = categorical_features + non_categorical_features
        if "is_mixed" in data.columns:
            features.append("is_mixed")

        label_columns = ["label"]

        for col in categorical_features:
            data[col] = data[col].astype("category")

        if self.model_params["balance_motifs_in_train_set"]:
            data = self.balance_motifs(data)

        data.loc[:, "label"] = data["label"] == "TP"

        if not (data.shape[0] > desired_n):
            logger.error("data size smaller than requested")
            return

        # split data to train and test
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data, data[label_columns], test_size=0.33, random_state=42
        # )
        X_train, X_test, y_train, y_test = self.train_test_split_head_tail(data, label_columns)

        xgb_classifier = xgb.XGBClassifier(
            n_estimators=int(self.model_params["forest_size"]),
            objective="multi:softprob",
            tree_method="hist",
            eta=self.model_params["eta"],
            max_depth=self.model_params["max_depth"],
            n_jobs=self.model_params["n_jobs"],
            enable_categorical=True,
            subsample=self.model_params["subsample"],
            num_class=2,
        )

        xgb_classifier.fit(X_train[features], y_train["label"])

        # save classifier and data, generate plots for report
        self.save_model_and_data(
            xgb_classifier,
            X_test,
            y_test,
            X_train,
            y_train,
        )

        return self


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


def create_filters_file(filters_file_path):
    # load/define filters
    edist_filter = "X_EDIST <= 5"
    trim_ends = "X_INDEX > 12 and X_INDEX < (X_LENGTH - 12)"
    filters_hueristic = {
        "no_filter": f"X_SCORE >= 0",
        "edit_distance_5": f"X_SCORE >= 0 and {edist_filter}",
        "BQ60": f"X_SCORE >= 6 and {edist_filter}",
        "BQ80": f"X_SCORE >= 7.9 and {edist_filter}",
        "CSKP": f"X_SCORE >= 10 and {edist_filter}",
        "BQ80_trim_ends": f"X_SCORE >= 7.9 and {edist_filter} and {trim_ends}",
        "CSKP_trim_ends": f"X_SCORE >= 10 and {edist_filter} and {trim_ends}",
        "BQ60_mixed_only": f"X_SCORE >= 6 and {edist_filter} and is_mixed",
        "BQ80_mixed_only": f"X_SCORE >= 7.9 and {edist_filter} and is_mixed",
        "CSKP_mixed_only": f"X_SCORE >= 10 and {edist_filter} and is_mixed",
        "BQ80_trim_ends_mixed_only": f"X_SCORE >= 7.9 and {edist_filter} and {trim_ends} and is_mixed",
        "CSKP_trim_ends_mixed_only": f"X_SCORE >= 10 and {edist_filter} and {trim_ends} and is_mixed",
    }

    xgb_filters = {f"xgb_qual_{q}": f"XGB_qual_1 >= {q}" for q in range(0, 29)}

    filters_dict = {
        **filters_hueristic,
        **xgb_filters,
    }

    with open(filters_file_path, "w") as f:
        json.dump(filters_dict, f, indent=6)


def inference_on_dataframe(xgb_classifier, X, y):
    cls_features = xgb_classifier.feature_names_in_
    probs = xgb_classifier.predict_proba(X[cls_features])
    predictions = xgb_classifier.predict(X[cls_features])
    quals = -10 * np.log10(1 - probs)
    # labels = np.unique(y["label"])
    predictions_df = pd.DataFrame(y)
    labels = np.unique(y["label"].astype(int))
    for label in labels:
        predictions_df[f"XGB_prob_{label}"] = probs[:, label]
        predictions_df[f"XGB_qual_{label}"] = quals[:, label]
        predictions_df[f"XGB_prediction_{label}"] = predictions[:, label]
    df = pd.concat([X, predictions_df.drop(["label"], axis="columns", inplace=False)], axis=1)

    df_tp = df.query(f"label == True")
    df_fp = df.query(f"label == False")

    fprs = {}
    recalls = {}
    max_score = -1
    for label in labels:
        fprs[label] = []
        recalls[label] = []
        score = f"XGB_qual_{label}"
        max_score = np.max((int(np.ceil(df[score].max())), max_score))
        gtr = df["label"] == label
        fprs_, recalls_ = precision_recall_curve(df[score], max_score=max_score, y_true=gtr)
        fprs[label].append(fprs_)
        recalls[label].append(recalls_)

    return df, df_tp, df_fp, labels, max_score, cls_features, fprs, recalls


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

    recall = dict()
    precision = dict()
    for xvar, querystr, namestr in (
        ("X_SCORE", "X_SCORE>-1", "X_SCORE"),
        ("X_SCORE", "is_mixed == True & X_SCORE>-1", "X_SCORE_mixed"),
        (score, "X_SCORE>-1", score),
    ):
        thresholds = np.linspace(0, df[xvar].quantile(0.99), 100)
        recall[namestr] = dict()
        precision[namestr] = dict()
        for thresh in thresholds:
            recall[namestr][thresh] = (df_tp.query(querystr)[xvar] > thresh).sum() / df_tp.shape[0]
            total_fp = (df_fp.query(querystr)[xvar] >= thresh).sum()
            total_tp = (df_tp.query(querystr)[xvar] >= thresh).sum()
            precision[namestr][thresh] = total_tp / (total_tp + total_fp)

    # set_pyplot_defaults()
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for label in recall.keys():
        # plt.plot(recall[label].values(), precision[label].values(), "k")
        auc = -np.trapz(list(precision[label].values()), list(recall[label].values()))
        plt.plot(
            recall[label].values(),
            precision[label].values(),
            "-o",
            # c=list(precision[label].keys()),
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

    return


def simulate_LoD(single_sub_regions, cram_stats_file, sampling_rate, df, filters_file):
    # apply filters to data
    with open(filters_file) as f:
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
    with open(single_sub_regions) as fh:
        for line in fh:
            if not line.startswith("@") and not line.startswith("#"):
                spl = line.rstrip().split("\t")
                n_bases_in_region += int(spl[2]) - int(spl[1])

    # get coverage statistics
    (
        mean_coverage,
        ratio_of_reads_over_mapq,
        ratio_of_bases_in_coverage_range,
        min_coverage_for_fp,
        coverage_of_max_percentile,
    ) = read_effective_coverage_from_sorter_json(cram_stats_file)

    # simulated LoD definitions and correction factors
    simulated_signature_size = 10_000
    simulated_coverage = 30
    effective_signature_bases_covered = (
        simulated_coverage * simulated_signature_size
    )  # * ratio_of_reads_over_mapq * read_filter_correction_factor
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
    sensitivity_at_lod = 0.90
    specificity_at_lod = 0.99

    df_summary = pd.concat(
        (
            (df_fp[list(filters.keys())].sum() / n_noise_reads).rename("FP"),
            (df_tp[list(filters.keys())].sum() / n_signal_reads * ratio_filtered_prior_to_featuremap).rename("TP"),
        ),
        axis=1,
    )
    # save intermediate output files
    # save df_summary, generate a method with input: df_summary, signature parameters,
    # returns df_mrd_sim and the filter that corresponds to best LoD
    df_summary = df_summary.assign(precision=df_summary["TP"] / (df_summary["FP"] + df_summary["TP"]))

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
        .clip(min=2)  # TODO: make it a parameter
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
    lod_label = f"LoD @ {specificity_at_lod*100:.0f}% specificity, {sensitivity_at_lod*100:.0f}% sensitivity (estimated)\nsignature size {simulated_signature_size}, {simulated_coverage}x coverage"
    c_lod = f"LoD_{sensitivity_at_lod*100:.0f}"

    return df_mrd_sim, lod_label, c_lod, filters


def plot_LoD(df_mrd_sim, lod_label, c_lod, filters, title="", output_filename=""):
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
        import matplotlib.colors as colors  # https://matplotlib.org/stable/tutorials/colors/colormapnorms.html

        plt.plot(
            df_tmp["read_retention_ratio"],
            df_tmp["residual_snv_rate"],
            c="k",
            alpha=0.3,
        )
        best_lod = df_tmp[c_lod].min()
        mini, maxi = (
            7e-7,
            2e-5,
        )  # or use different method to determine the minimum and maximum to use
        # norm = colors.PowerNorm(gamma=0.2,vmin=mini, vmax=maxi) #plt.Normalize(mini, maxi)
        sc = plt.scatter(
            df_tmp["read_retention_ratio"],
            df_tmp["residual_snv_rate"],
            c=df_tmp[c_lod],
            marker=marker,
            edgecolor=edgecolor,
            # cmap="viridis",
            label=label + ", best LoD: {:.1E}".format(best_lod).replace("E-0", "E-"),
            s=markersize,
            zorder=markersize,
            # norm=colors.PowerNorm(gamma=0.2,vmin=mini, vmax=maxi)
        )
        # sc.set_clim(mini,maxi)

    plt.xlabel("Read retention ratio on HOM SNVs", fontsize=24)
    plt.ylabel("Residual SNV rate", fontsize=24)
    plt.yscale("log")
    title_handle = plt.title(title, fontsize=24)
    legend_handle = plt.legend(fontsize=10, fancybox=True, framealpha=0.95)

    def fmt(x, pos):
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        return r"${} \times 10^{{{}}}$".format(a, b)

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
    return


def plot_confusion_matrix(
    df: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
    cm = confusion_matrix(df[f"XGB_prediction_1"], df["label"])
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

    return


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

    return


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

    return


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
    return


def plot_ML_qual_hist(
    labels_dict,
    df,
    max_score,
    title: str = "",
    output_filename: str = None,
    fs=14,
):
    score = f"XGB_qual_1"

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

    return


def plot_qual_per_feature(
    labels_dict,
    df,
    max_score,
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

    return


def get_mixed_data(df):
    df_mixed_cs = df[(df["is_mixed"]) & (df["cycle_skip_status"] == "cycle-skip")]
    df_mixed_non_cs = df[(df["is_mixed"]) & (df["cycle_skip_status"] != "cycle-skip")]
    df_non_mixed_non_cs = df[(~df["is_mixed"]) & (df["cycle_skip_status"] != "cycle-skip")]
    df_non_mixed_cs = df[(~df["is_mixed"]) & (df["cycle_skip_status"] == "cycle-skip")]
    return df_mixed_cs, df_mixed_non_cs, df_non_mixed_non_cs, df_non_mixed_cs


def get_fpr_recalls_mixed(df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs):
    score = f"XGB_qual_1"
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
    fs=14,
):
    score = f"XGB_qual_1"

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
        # plt.ylim([0, 120000])
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

    return


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

    return


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

    return


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
    with open(params_file, "r") as f:
        params = json.load(f)

    (
        df,
        df_tp,
        df_fp,
        labels,
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
    df_mrd_sim, lod_label, c_lod, filters = simulate_LoD(
        params["single_sub_regions"], cram_stats_file, sampling_rate, df, filters_file
    )

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
        df,
        max_score,
        title=f"{params['data_name']}\nqual per ",
        output_filename=output_qual_per_feature,
    )
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
        ) = get_fpr_recalls_mixed(df_mixed_cs, df_mixed_non_cs, df_non_mixed_cs, df_non_mixed_non_cs)

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

    return
