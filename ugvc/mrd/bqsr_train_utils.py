from __future__ import annotations

import json
import os
from os.path import basename, isfile
from os.path import join as pjoin

import joblib
import numpy as np
import pandas as pd
import pysam
import sklearn
import xgboost as xgb
from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.featuremap_utils import FeatureMapFields
from ugvc.srsnv.srsnv_utils import balance_df_according_motif, precision_recall_curve
from ugvc.utils.consts import FileExtension
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json
from ugvc.utils.misc_utils import exec_command_list

default_xgboost_model_params = {
    "balance_motifs_in_train_set": True,
    "sample_size": 50000000,
    "n_estimators": 100,
    "objective": "multi:softprob",
    "tree_method": "hist",
    "eta": 0.1,
    "max_depth": 10,
    "n_jobs": 16,
    "subsample": 0.35,
    "num_class": 2,
    "enable_categorical": True,
}
default_numerical_features = [
    FeatureMapFields.X_SCORE.value,
    FeatureMapFields.X_EDIST.value,
    FeatureMapFields.X_LENGTH.value,
    FeatureMapFields.X_INDEX.value,
    FeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
]
default_categorical_features = [
    FeatureMapFields.IS_CYCLE_SKIP.value,
    FeatureMapFields.TRINUC_CONTEXT.value,
    FeatureMapFields.ALT.value,
]


def prepare_featuremap_for_model(
    workdir: str,
    input_featuremap_vcf: str,
    training_set_size: int,
    test_set_size: int = None,
    sp: SimplePipeline = None,
    regions_file: str = None,
    balance_motifs: bool = True,
    sorter_json_stats_file: str = None,
    read_effective_coverage_from_sorter_json_kwargs: dict = None,
    keep_temp_file: bool = False,
    random_seed: bool = None,
) -> (str, int):
    """
    Prepare a featuremap for training. This function takes an input featuremap and creates two downsampled dataframes,
    one for training (size training_set_size) and one for testing (size test_set_size). The first training_set_size
    entries in the input featuremap are used for training and the last test_set_size entries are used for testing.

    test_set_size in practice could vary because statistical sampling is used, train_set_size is exact unless dataset
    is smaller than training_set_size+test_set_size.


    Parameters
    ----------
    workdir : str
        Working directory to which outputs will be written
    input_featuremap_vcf : str
        Path to input featuremap
    training_set_size : int
        Size of training set to create, must be at larger than 0
    test_set_size : int
        Size of test set to create, must be at larger than 0
    sp : SimplePipeline, optional
        SimplePipeline object to use for printing and running commands
    regions_file : str, optional
        Path to regions file, by default None
    sorter_json_stats_file : str, optional
        Path to cram stats file, by default None
    read_effective_coverage_from_sorter_json_kwargs : dict, optional
        kwargs for read_effective_coverage_from_sorter_json, by default None
    keep_temp_file: bool, optional
        Whether to keep the intersected featuremap file created prior to sampling, by default False

    Returns
    -------
    str
        Path to sampled featuremap
    int
        Number of entries in intersected featuremap

    """

    # check inputs and define paths
    os.makedirs(workdir, exist_ok=True)
    assert isfile(input_featuremap_vcf), f"input_featuremap_vcf {input_featuremap_vcf} not found"
    assert input_featuremap_vcf.endswith(FileExtension.VCF_GZ.value)
    assert training_set_size > 0, f"training_set_size must be > 0, got {training_set_size}"
    assert test_set_size is None or (test_set_size > 0), f"test_set_size must be > 0, got {test_set_size}"
    if not read_effective_coverage_from_sorter_json_kwargs:
        read_effective_coverage_from_sorter_json_kwargs = {}
    # make sure X_READ_COUNT is in the INFO fields in the header
    with pysam.VariantFile(input_featuremap_vcf) as fmap:
        assert FeatureMapFields.READ_COUNT.value in fmap.header.info
    intersect_featuremap_vcf = pjoin(
        workdir,
        basename(input_featuremap_vcf).replace(FileExtension.VCF_GZ.value, ".intersect.vcf.gz"),
    )
    downsampled_training_featuremap_vcf = pjoin(
        workdir,
        basename(intersect_featuremap_vcf).replace(FileExtension.VCF_GZ.value, ".training.downsampled.vcf.gz"),
    )
    downsampled_test_featuremap_vcf = pjoin(
        workdir,
        basename(intersect_featuremap_vcf).replace(FileExtension.VCF_GZ.value, ".test.downsampled.vcf.gz"),
    )
    assert not (isfile(intersect_featuremap_vcf)), f"intersect_featuremap_vcf {intersect_featuremap_vcf} already exists"
    assert not (
        isfile(downsampled_training_featuremap_vcf)
    ), f"sample_featuremap_vcf_sorted {downsampled_training_featuremap_vcf} already exists"

    # filter featuremap - intersect with bed file and require coverage in range
    commandstr = f"bcftools view {input_featuremap_vcf}"
    if sorter_json_stats_file:
        # get min and max coverage from cram stats file
        (_, _, _, min_coverage_for_fp, max_coverage_for_fp,) = read_effective_coverage_from_sorter_json(
            sorter_json_stats_file, **read_effective_coverage_from_sorter_json_kwargs
        )
        # filter out variants with coverage outside of min and max coverage
        commandstr = (
            commandstr
            + f" -i' (INFO/{FeatureMapFields.READ_COUNT.value} >= {min_coverage_for_fp}) "
            + f"&& (INFO/{FeatureMapFields.READ_COUNT.value} <= {max_coverage_for_fp}) '"
        )
    if regions_file:
        commandstr = commandstr + f" -R {regions_file} "
    commandstr = commandstr + f" -O z -o {intersect_featuremap_vcf}"
    commandslist = [commandstr, f"bcftools index -t {intersect_featuremap_vcf}"]
    exec_command_list(commandslist, sp)
    assert os.path.isfile(intersect_featuremap_vcf), f"failed to create {intersect_featuremap_vcf}"

    # count entries in intersected featuremap and determine downsampling rate
    total_size = training_set_size + test_set_size if test_set_size else training_set_size
    with pysam.VariantFile(intersect_featuremap_vcf) as fmap:
        featuremap_entry_number = 0
        for _ in enumerate(fmap.fetch()):
            featuremap_entry_number += 1
    if featuremap_entry_number < total_size:
        logger.warning(
            f"featuremap_entry_number={featuremap_entry_number} > training_set_size={training_set_size}"
            f"+ test_set_size={test_set_size if test_set_size else 0}"
            "\nbehavior is undefined",
        )
    # set sampling rate to be slightly higher than the desired training set size
    overhead_factor = 1.03  # 3% overhead to make sure we get the desired number of entries
    downsampling_rate = overhead_factor * (total_size) / featuremap_entry_number
    logger.info(
        f"training_set_size={training_set_size}, featuremap_entry_number = {featuremap_entry_number},"
        f"downsampling_rate {downsampling_rate}"
    )
    logger.info(
        f"test_set_size~={test_set_size}, featuremap_entry_number = {featuremap_entry_number}, "
        f"downsampling_rate {downsampling_rate}"
    )

    # random sample of intersected featuremap - sampled featuremap
    if random_seed is not None:
        np.random.seed(random_seed)
    if not balance_motifs:
        sampling_array = np.random.uniform(size=featuremap_entry_number) < downsampling_rate
        while sum(sampling_array) < min(total_size, featuremap_entry_number):
            sampling_array = np.random.uniform(size=featuremap_entry_number) < downsampling_rate
        record_counter = 0
        with pysam.VariantFile(input_featuremap_vcf) as vcf_in:
            # Add a comment to the output file indicating the subsampling
            header_train = vcf_in.header
            header_train.add_line(
                f"##datatype=training_set, subsampled {training_set_size}"
                f"variants from a total of {featuremap_entry_number}"
            )
            header_test = vcf_in.header
            header_test.add_line(
                f"##datatype=test_set, subsampled approximately {test_set_size}"
                f"variants from a total of {featuremap_entry_number}"
            )
            with pysam.VariantFile(
                downsampled_training_featuremap_vcf, "w", header=header_train
            ) as vcf_out_train, pysam.VariantFile(
                downsampled_test_featuremap_vcf, "w", header=header_train
            ) as vcf_out_test:
                for j, rec in enumerate(vcf_in.fetch()):
                    if sampling_array[j]:
                        # write the first training_set_size records to the training set
                        if record_counter < training_set_size:
                            vcf_out_train.write(rec)
                        # write the next test_set_size records to the test set
                        elif record_counter < total_size:
                            vcf_out_test.write(rec)
                        else:
                            break
                        record_counter += 1
    # create tabix index
    pysam.tabix_index(downsampled_training_featuremap_vcf, preset="vcf", force=True)
    pysam.tabix_index(downsampled_test_featuremap_vcf, preset="vcf", force=True)
    assert isfile(downsampled_training_featuremap_vcf), f"failed to create {downsampled_training_featuremap_vcf}"
    assert isfile(
        downsampled_training_featuremap_vcf + ".tbi"
    ), f"failed to create {downsampled_training_featuremap_vcf}.tbi"
    assert isfile(downsampled_test_featuremap_vcf), f"failed to create {downsampled_test_featuremap_vcf}"
    assert isfile(downsampled_test_featuremap_vcf + ".tbi"), f"failed to create {downsampled_test_featuremap_vcf}.tbi"

    # remove temp files
    if not keep_temp_file:
        if isfile(intersect_featuremap_vcf):
            os.remove(intersect_featuremap_vcf)
        if isfile(intersect_featuremap_vcf + ".tbi"):
            os.remove(intersect_featuremap_vcf + ".tbi")
    if test_set_size is None:  # an empty vcf file should have been created
        os.remove(downsampled_test_featuremap_vcf)
        os.remove(downsampled_test_featuremap_vcf + ".tbi")

    return downsampled_training_featuremap_vcf, downsampled_test_featuremap_vcf


class BQSRTrain:  # pylint: disable=too-many-instance-attributes
    MIN_TEST_SIZE = 10000
    MIN_TRAIN_SIZE = 100000

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        out_path: str,
        out_basename: str,
        tp_featuremap: str,
        fp_featuremap: str,
        numerical_features: list[str] = None,
        categorical_features: list[str] = None,
        sorter_json_stats_file: str = None,
        tp_regions_bed_file: str = None,
        fp_regions_bed_file: str = None,
        flow_order: str = DEFAULT_FLOW_ORDER,
        model_parameters: dict | str = None,
        classifier_class=xgb.XGBClassifier,
        train_set_size: int = 100000,
        test_set_size: int = 10000,
        balance_motifs_in_train_set: bool = True,
        simple_pipeline: SimplePipeline = None,
    ):
        """
        Train a classifier on FP and TP featuremaps

        Parameters
        ----------
        out_path : str
            Path to output directory
        out_basename : str
            Basename for output files
        tp_featuremap : str
            Path to featuremap of true positives (generally homozygous SNVs)
        fp_featuremap : str
            Path to featuremap of false positives (generally single substitutions)
        numerical_features : list[str], optional
            List of numerical features to use, default (None): [X_SCORE,X_EDIST,X_LENGTH,X_INDEX,MAX_SOFTCLIP_LENGTH]
        categorical_features : list[str], optional
            List of categorical features to use, default (None): [IS_CYCLE_SKIP,TRINUC_CONTEXT,ALT]
        tp_regions_bed_file : str, optional
            Path to bed file of regions to use for true positives
        fp_regions_bed_file : str, optional
            Path to bed file of regions to use for false positives
        flow_order : str, optional
            Flow order, by default TGCA
        model_parameters : dict, optional
            Parameters for classifier, by default None (default_xgboost_model_params)
            If a string is given then it is assumed to be a path to a json file with the parameters
        sorter_json_stats_file : str, optional
            Path to sorter stats json file, if not given then the normalized error rates cannot be calculated
        classifier_class : xgb.XGBClassifier, optional
            class from which classifier will be instantiated, by default xgb.XGBClassifier
            sklearn.base.is_classifier() must return True for this class
        balance_motifs_in_train_set : bool, optional
            Whether to balance the motifs (trinuc context) in the training set, by default True
            Recommended in order to avoid the motif distribution of germline variants being learned (data leak)
        simple_pipeline : SimplePipeline, optional
            SimplePipeline object to use for printing and running commands, by default None

        Raises
        ------
        ValueError
            If test_set_size < MIN_TEST_SIZE or train_set_size < MIN_TRAIN_SIZE

        """
        # default values
        model_params = model_params or default_xgboost_model_params
        categorical_features = categorical_features or default_categorical_features
        numerical_features = numerical_features or default_numerical_features

        # determine output paths
        os.makedirs(out_path, exist_ok=True)
        if len(out_basename) > 0 and not out_basename.endswith("."):
            out_basename += "."
        self.out_path = out_path
        self.out_basename = out_basename
        (
            self.model_save_path,
            self.X_test_save_path,
            self.y_test_save_path,
            self.X_train_save_path,
            self.y_train_save_path,
            self.params_save_path,
        ) = self._get_file_paths()

        # set up classifier
        assert sklearn.base.is_classifier(
            classifier_class()
        ), "classifier_class must be a classifier - sklearn.base.is_classifier() must return True"
        self.classifier_type = type(classifier_class).__name__
        if isfile(model_parameters) and model_parameters.endswith(".json"):
            with open(model_parameters, "r", encoding="utf-8") as f:
                model_params = json.load(f)
        elif model_parameters is None:
            model_params = default_xgboost_model_params
        self.model_parameters = model_params
        self.classifier = classifier_class(**self.model_parameters)
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.balance_motifs_in_train_set = balance_motifs_in_train_set

        # save input data file paths
        assert isfile(tp_featuremap), f"tp_featuremap {tp_featuremap} not found"
        self.hom_snv_featuremap = tp_featuremap
        assert isfile(fp_featuremap), f"fp_featuremap {fp_featuremap} not found"
        self.single_substitution_featuremap = fp_featuremap
        if tp_regions_bed_file:
            assert isfile(tp_regions_bed_file), f"tp_regions_bed_file {tp_regions_bed_file} not found"
        self.tp_regions_bed_file = tp_regions_bed_file
        if fp_regions_bed_file:
            assert isfile(fp_regions_bed_file), f"fp_regions_bed_file {fp_regions_bed_file} not found"
        self.fp_regions_bed_file = fp_regions_bed_file
        if sorter_json_stats_file:
            assert isfile(sorter_json_stats_file), f"sorter_json_stats_file {sorter_json_stats_file} not found"
        self.sorter_json_stats_file = sorter_json_stats_file

        # misc
        self.flow_order = flow_order
        self.sp = simple_pipeline

        # set and check train and test sizes
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size

        if self.train_set_size < BQSRTrain.MIN_TEST_SIZE:
            msg = f"test_size must be >= {BQSRTrain.MIN_TEST_SIZE}"
            logger.error(msg)
            raise ValueError(msg)
        if self.test_set_size < BQSRTrain.MIN_TRAIN_SIZE:
            msg = f"train_size must be >= {BQSRTrain.MIN_TRAIN_SIZE}"
            logger.error(msg)
            raise ValueError(msg)

    def _get_file_paths(self):
        return (
            pjoin(f"{self.out_path}, {self.out_basename}model.joblib"),
            pjoin(f"{self.out_path}, {self.out_basename}X_test.parquet"),
            pjoin(f"{self.out_path}, {self.out_basename}y_test.parquet"),
            pjoin(f"{self.out_path}, {self.out_basename}X_train.parquet"),
            pjoin(f"{self.out_path}, {self.out_basename}y_train.parquet"),
            pjoin(f"{self.out_path}, {self.out_basename}params.json"),
        )

    def balance_motifs(self, data):
        tp_df = data[data["label"] == "TP"]
        fp_df = data[data["label"] == "FP"]

        balanced_tps = balance_df_according_motif(tp_df, "ref_alt_motif", 0)
        data = pd.concat([balanced_tps, fp_df], ignore_index=True)

        return data

    # prepare featuremaps for training
    def prepare_featuremaps_for_model(self):
        featuremaplist = [self.single_substitution_featuremap, self.hom_snv_featuremap]
        regionslist = [self.fp_regions_bed_file, self.tp_regions_bed_file]
        statlist = [self.sorter_json_stats_file, None]

        # training set of size will include n TP and n FP where n ~ 0.5*(test_size+train_size)
        featuremaps_for_training = []
        total_n_list = []
        for featuremap, regions_file, sorter_json_stats_file in zip(featuremaplist, regionslist, statlist):
            featuremap_for_training, total_n = prepare_featuremap_for_model(
                self.sp,
                self.out_path,
                featuremap,
                train_set_size=self.train_set_size,
                test_set_size=self.test_set_size,
                tp_regions_bed_file=self.tp_regions_bed_file,
                fp_regions_bed_file=self.fp_regions_bed_file,
                sorter_json_stats_file=self.sorter_json_stats_file,
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

    def train_test_split_head_tail(self, data, label_columns):
        df_tp = data.query("label")
        df_fp = data.query("not label")

        n_train = self.train_set_size // 2
        n_test = self.test_set_size // 2
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
        # load training data
        data = self.extract_data_from_featuremap_local()
        logger.info(f"Labels in data: \n{data['label'].value_counts()}")

        # model init and train
        motif_only_features = self.model_params["motif_only_features"]
        categorical_features = self.model_params["categorical_features"] + motif_only_features
        non_categorical_features = self.model_params["non_categorical_features"]

        features = categorical_features + non_categorical_features

        label_columns = ["label"]

        for col in categorical_features:
            data[col] = data[col].astype("category")

        if self.model_params["balance_motifs_in_train_set"]:
            data = self.balance_motifs(data)

        data.loc[:, "label"] = data["label"] == "TP"

        # if not (data.shape[0] > desired_n):
        #     logger.error("data size smaller than requested")
        #     return

        # split data to train and test
        # X_train, X_test, y_train, y_test = train_test_split(
        #     data, data[label_columns], test_size=0.33, random_state=42
        # )
        X_train, X_test, y_train, y_test = self.train_test_split_head_tail(data, label_columns)

        # xgb_classifier = xgb.XGBClassifier(
        #     n_estimators=int(self.model_params["forest_size"]),
        #     objective="multi:softprob",
        #     tree_method="hist",
        #     eta=self.model_params["eta"],
        #     max_depth=self.model_params["max_depth"],
        #     n_jobs=self.model_params["n_jobs"],
        #     enable_categorical=True,
        #     subsample=self.model_params["subsample"],
        #     num_class=2,
        # )

        self.classifier.fit(X_train[features], y_train["label"])

        # save classifier and data, generate plots for report
        self.save_model_and_data(
            self.classifier,
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
        "no_filter": "X_SCORE >= 0",
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

    df_tp = df.query("label == True")
    df_fp = df.query("label == False")

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
