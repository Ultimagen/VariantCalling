from __future__ import annotations

import json
import os
from enum import Enum
from os.path import basename, isfile
from os.path import join as pjoin

import joblib
import numpy as np
import pandas as pd
import pysam
import xgboost as xgb
from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugvc.srsnv.srsnv_utils import balance_df_according_motif, precision_recall_curve
from ugvc.utils.consts import FileExtension
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json
from ugvc.utils.misc_utils import exec_command_list

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


def prepare_featuremap_for_training(
    workdir: str,
    input_featuremap_vcf: str,
    training_set_size: int,
    sp: SimplePipeline = None,
    regions_file: str = None,
    cram_stats_file: str = None,
    read_effective_coverage_from_sorter_json_kwargs: dict = None,
) -> (str, int):
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
    int
        Number of entries in intersected featuremap

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
