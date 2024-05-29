from __future__ import annotations

import json
import os
from collections import defaultdict
from os.path import basename, isfile
from os.path import join as pjoin

import joblib
import numpy as np
import pandas as pd
import pysam
import sklearn
import xgboost as xgb
from simppl.simple_pipeline import SimplePipeline
from datetime import datetime
from typing import Any

from ugvc import logger
from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.featuremap_utils import FeatureMapFields, filter_featuremap_with_bcftools_view
from ugvc.mrd.mrd_utils import featuremap_to_dataframe
from ugvc.mrd.srsnv_inference_utils import get_quality_interpolation_function
from ugvc.mrd.srsnv_plotting_utils import create_report
from ugvc.utils.consts import FileExtension
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json

default_xgboost_model_params = {
    "n_estimators": 100,
    "objective": "multi:softprob",
    "tree_method": "hist",
    "eta": 0.15,
    "max_depth": 10,
    "n_jobs": 16,
    "subsample": 0.9,
    "num_class": 2,
    "enable_categorical": True,
    "colsample_bytree": 0.65,
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
    FeatureMapFields.ALT.value,
    FeatureMapFields.REF.value,
    FeatureMapFields.PREV_1.value,
    FeatureMapFields.PREV_2.value,
    FeatureMapFields.PREV_3.value,
    FeatureMapFields.NEXT_1.value,
    FeatureMapFields.NEXT_2.value,
    FeatureMapFields.NEXT_3.value,
]

CHROM_SIZES = {
    'chr1': 248956422,
    'chr2': 242193529,
    'chr3': 198295559,
    'chr4': 190214555,
    'chr5': 181538259,
    'chr6': 170805979,
    'chr7': 159345973,
    'chr8': 145138636,
    'chr9': 138394717,
    'chr11': 135086622,
    'chr10': 133797422,
    'chr12': 133275309,
    'chr13': 114364328,
    'chr14': 107043718,
    'chr15': 101991189,
    'chr16': 90338345,
    'chr17': 83257441,
    'chr18': 80373285,
    'chr20': 64444167,
    'chr19': 58617616,
    'chr22': 50818468,
    'chr21': 46709983
}

def partition_into_folds(series_of_sizes, k_folds, alg='greedy'):
    """Returns a partition of the indices of the series series_of_sizes 
    into k_fold groups whose total size is approximately the same.
    Returns a dictionary that maps the indices (keys) of series_of_sizes into
    the corresponding fold number (partition). 
    
    If series_of_sizes is a series, then the list-of-lists partitions below satisfies that:
    [series_of_sizes.loc[partitions[k]].sum() for k in range(k_folds)] 
    are approximately equal. Conversely, 
    series_of_sizes.groupby(indices_to_folds).sum()
    are approximately equal. 

    Arguments:
        - series_of_sizes [pd.Series]: a series of indices and their corresponding sizes. 
        - k_folds [int]: the number of folds into which series_of_sizes should be partitioned.
        - alg ['greedy']: the algorithm used. For the time being only the greedy algorithm
            is implemented. 
    Returns: 
        - indices_to_folds [dict]: a dictionary that maps indices to the corresponding 
            fold numbers.
    """
    assert alg=='greedy', 'Only greedy algorithm implemented at this time'
    series_of_sizes = series_of_sizes.sort_values(ascending=False)
    partitions = [[] for _ in range(k_folds)] # an empty partition
    partition_sums = np.zeros(k_folds) # The running sum of partitions
    for idx, s in series_of_sizes.items():
        min_fold = partition_sums.argmin()
        partitions[min_fold].append(idx)
        partition_sums[min_fold] += s
        
    #return partitions
    indices_to_folds = [[i for i, prtn in enumerate(partitions) if idx in prtn][0] for idx in series_of_sizes.index]
    return pd.Series(indices_to_folds, index=series_of_sizes.index).to_dict()

def prob_to_phred(prob, eps=1e-8):
    """Transform probabilities to phred scores. 
    Arguments:
    - prob [np.ndarray]: array of probabilities
    - eps [float]: cutoff value (phred values can have maximum value of -10*np.log10(eps)
                   which for the default value 1e-8 means phred of 80)
    """
    return (-10*np.log10(1-prob+eps))

def phred_to_prob(phred):
    """Transform phred scores to probabilities. 
    Arguments:
    - phred [np.ndarray]: array of phred scores
    """
    return 1.0 - 10.0**(-phred/10)

def k_fold_predict_proba(
        models, df, columns_for_training, k_folds, 
        kfold_col='fold_id', return_train=False, **kwargs
    ):
    """Predict k-folds CV.
    NOTE that on test set (datapoint that do not belong to any fold) and on train set
    (if return_train) there are k_folds (or k_fold-1) models which can be used for 
    prediction. The function averages phred scores of the predictions of all theses
    models. 

    Arguments:
    - models: a list of k_folds models (one for each fold)
    - df: the data dataframe
    - kfold_col [str]: name of folder that contains fold numbers. If nan/negative then not
                        in any training fold (only test) 
    - columns_for_training [list]: names of columns used for training the model. 
    - return_train [bool]: If true, return also train values (probabilities when 
                           model is evaluated on the folds used for training it).
    """
    test_cond = df[kfold_col].isna()
    preds = pd.Series(0, index=df.index, name='test')
    all_val_cond = test_cond.copy() # for keeping track of all "test" indices
    if return_train:
        preds_train = pd.Series(0, index=df.index, name='train')
        preds_train_norm = (k_folds-1) if k_folds > 1 else 1
        all_train_cond = pd.Series(False, index=df.index) # for keeping track of all "train" indices
    for k in range(k_folds):
        val_cond = (df[kfold_col]==k)
        all_val_cond = np.logical_or(all_val_cond, val_cond)
        if val_cond.any():
            preds[val_cond] = models[k].predict_proba(df.loc[val_cond, columns_for_training], **kwargs)[:,1]
        if test_cond.any():
            test_scores = prob_to_phred(models[k].predict_proba(df.loc[test_cond, columns_for_training], **kwargs)[:,1])
            preds[test_cond] += test_scores/k_folds
        if return_train:
            train_cond = np.logical_and(~val_cond, ~test_cond)
            all_train_cond = np.logical_or(all_train_cond, train_cond)
            if train_cond.any():
                train_scores = prob_to_phred(models[k].predict_proba(df.loc[train_cond, columns_for_training], **kwargs)[:,1])
                preds_train[train_cond] += train_scores/preds_train_norm
    
    preds[test_cond] = phred_to_prob(preds[test_cond])
    preds[~all_val_cond] = np.nan 
    if return_train:
        preds_train = phred_to_prob(preds_train)
        preds_train[~all_train_cond] = np.nan
        preds = pd.concat((preds, preds_train), axis=1)

    return preds

# pylint:disable=too-many-arguments
def prepare_featuremap_for_model(
    workdir: str,
    input_featuremap_vcf: str,
    train_set_size: int,
    test_set_size: int = 0,
    k_folds : int = 1, 
    regions_file: str = None,
    balanced_sampling_info_fields: list[str] = None,
    sorter_json_stats_file: str = None,
    pre_filter_bcftools_include: str = None,
    read_effective_coverage_from_sorter_json_kwargs: dict = None,
    keep_temp_file: bool = False,
    rng: Any = None,
    sp: SimplePipeline = None,
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
    train_set_size : int
        Size of training set to create, must be at larger than 0
    test_set_size : int
        Size of test set to create, must be at larger than 0
    regions_file : str, optional
        Path to regions file, by default None
    sorter_json_stats_file : str, optional
        Path to cram stats file, by default None
    pre_filter_bcftools_include: str, optional
        bcftools include filter to apply as part of a "bcftools view <vcf> -i 'pre_filter_bcftools_include'"
        before sampling, by default None
    read_effective_coverage_from_sorter_json_kwargs : dict, optional
        kwargs for read_effective_coverage_from_sorter_json, by default None
    keep_temp_file: bool, optional
        Whether to keep the intersected featuremap file created prior to sampling, by default False
    rng : Any, optional
        Random number Generator. If None, initialize a generator with system time.
    balanced_sampling_info_fields: list[str], optional
        List of info fields to balance the TP data by, default None (do not balance)
    sp : SimplePipeline, optional
        SimplePipeline object to use for printing and running commands


    Returns
    -------
    downsampled_training_featuremap_vcf, str
        Downsampled training featuremap
    downsampled_test_featuremap_vcf, str
        Downsampled test featuremap
    int
        Number of entries in intersected featuremap

    """

    # check inputs and define paths
    os.makedirs(workdir, exist_ok=True)
    assert isfile(input_featuremap_vcf), f"input_featuremap_vcf {input_featuremap_vcf} not found"
    assert input_featuremap_vcf.endswith(FileExtension.VCF_GZ.value)
    assert train_set_size > 0, f"training_set_size must be > 0, got {train_set_size}"
    if k_folds == 1:
        assert test_set_size > 0, f"test_set_size must be > 0, got {test_set_size}"
    if not read_effective_coverage_from_sorter_json_kwargs:
        read_effective_coverage_from_sorter_json_kwargs = {}
    # make sure X_READ_COUNT is in the INFO fields in the header
    do_motif_balancing_in_tp = balanced_sampling_info_fields is not None
    with pysam.VariantFile(input_featuremap_vcf) as fmap:
        assert FeatureMapFields.READ_COUNT.value in fmap.header.info
        if do_motif_balancing_in_tp:
            for info_field in balanced_sampling_info_fields:
                assert info_field in fmap.header.info, f"INFO field {info_field} not found in header"
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

    logger.info(f"Running prepare_featuremap_for_model for input_featuremap_vcf={input_featuremap_vcf}")

    # filter featuremap - intersect with bed file, require coverage in range, apply pre_filter
    # get min and max coverage from cram stats file
    if sorter_json_stats_file:
        (_, _, _, min_coverage, max_coverage,) = read_effective_coverage_from_sorter_json(
            sorter_json_stats_file, **read_effective_coverage_from_sorter_json_kwargs
        )
    else:
        min_coverage = None
        max_coverage = None
    filter_featuremap_with_bcftools_view(
        input_featuremap_vcf=input_featuremap_vcf,
        intersect_featuremap_vcf=intersect_featuremap_vcf,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        regions_file=regions_file,
        bcftools_include_filter=pre_filter_bcftools_include,
        sp=sp,
    )

    # count entries in intersected featuremap and determine downsampling rate
    # count the number of entries with each combination of info fields to balance by if balanced_sampling_info_fields
    train_and_test_size = train_set_size + test_set_size
    balanced_sampling_info_fields_counter = defaultdict(int)
    with pysam.VariantFile(intersect_featuremap_vcf) as fmap:
        featuremap_entry_number = 0
        for record in fmap.fetch():
            featuremap_entry_number += 1
            if do_motif_balancing_in_tp:
                balanced_sampling_info_fields_counter[
                    tuple(record.info.get(info_field) for info_field in balanced_sampling_info_fields)
                ] += 1
    if featuremap_entry_number < train_and_test_size:
        logger.warning(
            "Requested training and test set size cannot be met - insufficient data"
            f"featuremap_entry_number={featuremap_entry_number} < training_set_size={train_set_size}"
            f"+ test_set_size={test_set_size if test_set_size else 0}"
        )
        train_set_size = np.floor(featuremap_entry_number * (train_set_size / train_and_test_size))
        test_set_size = featuremap_entry_number - train_set_size
        if k_folds > 1 and test_set_size > 0:
            msg = f'test_set_size should be 0 when using CV ({k_folds=}). Got {test_set_size=} for downsampled size'
            logger.error(msg)
            raise ValueError(msg)
        logger.warning(f"Set train_set_size to {train_set_size} and test_set_size to {test_set_size}")
    # set sampling rate to be slightly higher than the desired training set size
    overhead_factor = 1.03  # 3% overhead to make sure we get the desired number of entries
    downsampling_rate = overhead_factor * train_and_test_size / featuremap_entry_number
    logger.info(
        f"training_set_size={train_set_size}, featuremap_entry_number = {featuremap_entry_number},"
        f"downsampling_rate {downsampling_rate}"
    )
    # TODO: add:  if k_folds > 1:
    logger.info(
        f"test_set_size~={test_set_size}, featuremap_entry_number = {featuremap_entry_number}, "
        f"downsampling_rate {downsampling_rate}"
    )

    # random sample of intersected featuremap - sampled featuremap
    if rng is None:
        random_seed = int(datetime.now().timestamp())
        rng = np.random.default_rng(seed=random_seed)
        logger.info(f'Initializing random numer generator with {random_seed=}')
    if do_motif_balancing_in_tp:
        # determine sampling rate per group (defined by a combination of info fields)
        number_of_groups = len(balanced_sampling_info_fields_counter)
        train_and_test_size_per_group = train_and_test_size // number_of_groups
        downsampling_rate = {
            group_key: (train_and_test_size_per_group / group_total_entries_in_featuremap)
            for group_key, group_total_entries_in_featuremap in balanced_sampling_info_fields_counter.items()
        }
        for group_key, group_downsampling_rate in downsampling_rate.items():
            logger.debug(f"downsampling_rate for {group_key} = {group_downsampling_rate}")
            if group_downsampling_rate > 1:
                logger.warning(
                    f"downsampling_rate for {group_key} = {group_downsampling_rate} > 1, "
                    "result will contain less data than expected"
                )
            downsampling_rate[group_key] = min(1, group_downsampling_rate)

        record_counter = 0
        with pysam.VariantFile(intersect_featuremap_vcf) as vcf_in:
            # Add a comment to the output file indicating the subsampling
            header_train = vcf_in.header
            header_train.add_line(
                f"##datatype=training_set, subsampled {train_set_size}"
                f"variants from a total of {featuremap_entry_number}"
                f", balanced by {balanced_sampling_info_fields}"
            )
            # TODO: add:  if k_folds > 1:
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
                    if (
                        rng.uniform()
                        < downsampling_rate[
                            tuple(rec.info.get(info_field) for info_field in balanced_sampling_info_fields)
                        ]
                    ):
                        # write the first training_set_size records to the training set
                        if record_counter < train_set_size:
                            vcf_out_train.write(rec)
                        # write the next test_set_size records to the test set
                        elif record_counter < train_and_test_size:
                            vcf_out_test.write(rec)
                        else:
                            break
                        record_counter += 1
    else:
        sampling_array = rng.uniform(size=featuremap_entry_number + 1) < downsampling_rate
        while sum(sampling_array) < min(train_and_test_size, featuremap_entry_number):
            sampling_array = rng.uniform(size=featuremap_entry_number + 1) < downsampling_rate
        record_counter = 0
        with pysam.VariantFile(intersect_featuremap_vcf) as vcf_in:
            # Add a comment to the output file indicating the subsampling
            header_train = vcf_in.header
            header_train.add_line(
                f"##datatype=training_set, subsampled {train_set_size}"
                f"variants from a total of {featuremap_entry_number}"
            )
            # TODO: add:  if k_folds > 1:
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
                        if record_counter < train_set_size:
                            vcf_out_train.write(rec)
                        # write the next test_set_size records to the test set
                        elif record_counter < train_and_test_size:
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

    logger.info(
        f"Finished prepare_featuremap_for_model, outputting: "
        f"downsampled_training_featuremap_vcf={downsampled_training_featuremap_vcf}, "
        f"downsampled_test_featuremap_vcf={downsampled_test_featuremap_vcf}"
    )

    return (
        downsampled_training_featuremap_vcf,
        downsampled_test_featuremap_vcf,
        featuremap_entry_number,
        train_set_size,
        test_set_size,
    )


class SRSNVTrain:  # pylint: disable=too-many-instance-attributes
    MIN_TEST_SIZE = 10000
    MIN_TRAIN_SIZE = 100000

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        out_path: str,
        out_basename: str,
        tp_featuremap: str,
        fp_featuremap: str,
        train_set_size: int = MIN_TRAIN_SIZE,
        test_set_size: int = MIN_TEST_SIZE,
        k_folds: int = 1, 
        split_folds_by_chrom: bool = True,
        numerical_features: list[str] = None,
        categorical_features: list[str] = None,
        sorter_json_stats_file: str = None,
        tp_regions_bed_file: str = None,
        fp_regions_bed_file: str = None,
        flow_order: str = DEFAULT_FLOW_ORDER,
        model_params: dict | str = None,
        classifier_class=xgb.XGBClassifier,
        balanced_sampling_info_fields: list[str] = None,
        lod_filters: str = None,
        balanced_strand_adapter_version: str = None,
        pre_filter: str = None,
        random_seed: int = None,
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
        train_set_size : int, optional
            Size of training set, by default 100000 (not enough data for a good model, just for testing)
        test_set_size : int, optional
            Size of test set, by default 10000
        k_folds: int, optional
            The number of cross-validation folds to use. By default 1 (no CV). 
            If k_folds != 1, ignores test_set_size (there's no test set).
        numerical_features : list[str], optional
            List of numerical features to use, default (None): [X_SCORE,X_EDIST,X_LENGTH,X_INDEX,MAX_SOFTCLIP_LENGTH]
        categorical_features : list[str], optional
            List of categorical features to use, default (None): [IS_CYCLE_SKIP,TRINUC_CONTEXT_WITH_ALT]
        tp_regions_bed_file : str, optional
            Path to bed file of regions to use for true positives
        fp_regions_bed_file : str, optional
            Path to bed file of regions to use for false positives
        flow_order : str, optional
            Flow order, by default TGCA
        model_params : dict, optional
            Parameters for classifier, by default None (default_xgboost_model_params)
            If a string is given then it is assumed to be a path to a json file with the parameters
        sorter_json_stats_file : str, optional
            Path to sorter stats json file, if not given then the normalized error rates cannot be calculated
        classifier_class : xgb.XGBClassifier, optional
            class from which classifier will be instantiated, by default xgb.XGBClassifier
            sklearn.base.is_classifier() must return True for this class
        balanced_sampling_info_fields : list[str], optional
            List of info fields to balance the TP data by, default None (do not balance)
            Recommended in order to avoid the motif distribution of germline variants being learned (data leak)
        lod_filters : str, optional
            json file with a dict of format 'filter name':'query' for LoD simulation, by default None
        balanced_strand_adapter_version : str, optional
            adapter version, indicates if input featuremap is from balanced ePCR data, by default None
        pre_filter : str, optional
            bcftools include filter to apply as part of a "bcftools view <vcf> -i 'pre_filter'"
            before sampling, by default None
        random_seed : int, optional
            Random seed to use for sampling, by default None, If None, use system time. 
        simple_pipeline : SimplePipeline, optional
            SimplePipeline object to use for printing and running commands, by default None

        Raises
        ------
        ValueError
            If test_set_size < MIN_TEST_SIZE or train_set_size < MIN_TRAIN_SIZE
            If model_params is not a json file, dictionary or None

        """
        # default values
        model_params = model_params or default_xgboost_model_params
        categorical_features = categorical_features or default_categorical_features
        numerical_features = numerical_features or default_numerical_features

        # Check whether using cross validation: 
        assert k_folds > 0, f"k_folds should be > 0, got {k_folds=}"
        if k_folds == 1:
            self.use_CV = False
        else:
            self.use_CV = True
            logger.info(f"Using cross validation with {k_folds} folds")
        self.k_folds = k_folds
        self.split_folds_by_chrom = bool(split_folds_by_chrom)
        if self.split_folds_by_chrom:
            self.chroms_to_folds = partition_into_folds(
                pd.Series(CHROM_SIZES), 
                self.k_folds
            )

        # determine output paths
        os.makedirs(out_path, exist_ok=True)
        if len(out_basename) > 0 and not out_basename.endswith("."):
            out_basename += "."
        self.out_path = out_path
        self.out_basename = out_basename
        (
            self.model_save_path,
            self.featuremap_df_save_path,
            self.qual_test_save_path,
            self.params_save_path,
            self.test_mrd_simulation_dataframe_file,
            self.train_mrd_simulation_dataframe_file,
            self.test_statistics_h5_file,
            self.train_statistics_h5_file,
            self.test_statistics_json_file,
            self.train_statistics_json_file,
        ) = self._get_file_paths()

        # set up classifier
        assert sklearn.base.is_classifier(
            classifier_class()
        ), "classifier_class must be a classifier - sklearn.base.is_classifier() must return True"
        self.classifier_type = type(classifier_class).__name__
        if isinstance(model_params, str) and isfile(model_params) and model_params.endswith(".json"):
            with open(model_params, "r", encoding="utf-8") as f:
                model_params = json.load(f)
        elif model_params is None:
            model_params = default_xgboost_model_params
        elif isinstance(model_params, dict):
            pass
        else:
            raise ValueError("model_params should be json_file | dictionary | None")
        self.model_parameters = model_params
        self.classifiers = [classifier_class(**self.model_parameters) for _ in range(k_folds)]
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.balanced_sampling_info_fields = balanced_sampling_info_fields
        self.columns = self.numerical_features + self.categorical_features
        self.pre_filter = pre_filter
        if random_seed is None:
            random_seed = int(datetime.now().timestamp())
            logger.info(f'Initializing random numer generator with {random_seed=}')
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=self.random_seed)

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
        if isinstance(lod_filters, str) and isfile(lod_filters) and lod_filters.endswith(".json"):
            with open(lod_filters, "r", encoding="utf-8") as f:
                self.lod_filters = json.load(f)
        elif lod_filters is None:
            self.lod_filters = None
        elif isinstance(lod_filters, dict):
            self.lod_filters = lod_filters
        else:
            raise ValueError("lod_filters should be json_file | dictionary | None")

        self.flow_order = flow_order
        self.balanced_strand_adapter_version = balanced_strand_adapter_version
        self.sp = simple_pipeline

        # set and check train and test sizes
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        if self.use_CV:
            if test_set_size is not None:
                logger.info(f'Using cross-validation, Ignoring test_set_size (which was set to {test_set_size})')
            self.test_set_size = 0
        elif self.test_set_size < SRSNVTrain.MIN_TEST_SIZE:
            msg = f"test_size must be >= {SRSNVTrain.MIN_TEST_SIZE}"
            logger.error(msg)
            raise ValueError(msg)
        if self.train_set_size < SRSNVTrain.MIN_TRAIN_SIZE:
            msg = f"train_size must be >= {SRSNVTrain.MIN_TRAIN_SIZE}"
            logger.error(msg)
            raise ValueError(msg)

    def _get_file_paths(self):
        return (
            pjoin(f"{self.out_path}", f"{self.out_basename}model.joblib"),
            pjoin(f"{self.out_path}", f"{self.out_basename}featuremap_df.parquet"),
            pjoin(f"{self.out_path}", f"{self.out_basename}qual_test.parquet"),
            pjoin(f"{self.out_path}", f"{self.out_basename}params.json"),
            pjoin(f"{self.out_path}", f"{self.out_basename}test.df_mrd_simulation.parquet"),
            pjoin(
                f"{self.out_path}",
                f"{self.out_basename}train.df_mrd_simulation.parquet",
            ),
            pjoin(f"{self.out_path}", f"{self.out_basename}test.statistics.h5"),
            pjoin(f"{self.out_path}", f"{self.out_basename}train.statistics.h5"),
            pjoin(f"{self.out_path}", f"{self.out_basename}test.statistics.json"),
            pjoin(f"{self.out_path}", f"{self.out_basename}train.statistics.json"),
        )

    def get_params(self):
        return {
            "model_parameters": self.model_parameters,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "balanced_sampling_info_fields": self.balanced_sampling_info_fields,
            "hom_snv_featuremap": self.hom_snv_featuremap,
            "single_substitution_featuremap": self.single_substitution_featuremap,
            "tp_regions_bed_file": self.tp_regions_bed_file,
            "fp_regions_bed_file": self.fp_regions_bed_file,
            "sorter_json_stats_file": self.sorter_json_stats_file,
            "flow_order": self.flow_order,
            "train_set_size": self.train_set_size,
            "test_set_size": self.test_set_size,
            "fp_featuremap_entry_number": self.fp_featuremap_entry_number,
            "tp_featuremap_entry_number": self.tp_featuremap_entry_number,
            "fp_test_set_size": self.fp_test_set_size,
            "fp_train_set_size": self.fp_train_set_size,
            "tp_test_set_size": self.tp_test_set_size,
            "tp_train_set_size": self.tp_train_set_size,
            "lod_filters": self.lod_filters,
            "adapter_version": self.balanced_strand_adapter_version,
            "columns": self.columns,
            "featuremap_df_save_path": self.featuremap_df_save_path,
            "pre_filter": self.pre_filter,
            "random_seed": self.random_seed,
        }q

    def save_model_and_data(self):
        # save model
        joblib.dump(self.classifiers, self.model_save_path)
        # save data
        for data, savename in (
            (self.featuremap_df, self.featuremap_df_save_path),
            (self.qual_test, self.qual_test_save_path),
        ):
            data.to_parquet(savename)
        # save params
        params_to_save = self.get_params()
        with open(self.params_save_path, "w", encoding="utf-8") as f:
            json.dump(params_to_save, f)

    def add_predictions_to_featuremap_df(self, return_train=True):
        """ Add model predictions to self.featuremap_df. Use only if models were trained!
        """
        all_predictions = k_fold_predict_proba(
            self.classifiers, self.featuremap_df, self.columns, k_folds=self.k_folds, 
            return_train=return_train
        )
        datasets = ['test', 'train'] if return_train else ['test']
        for dataset in datasets:
            self.featuremap_df[f'ML_prob_1_{dataset}'] = all_predictions[dataset]
            self.featuremap_df[f'ML_prob_0_{dataset}'] = 1 - self.featuremap_df[f'ML_prob_1_{dataset}']
            self.featuremap_df[f'ML_qual_1_{dataset}'] = prob_to_phred(self.featuremap_df[f'ML_prob_1_{dataset}'])
            self.featuremap_df[f'ML_qual_0_{dataset}'] = prob_to_phred(self.featuremap_df[f'ML_prob_0_{dataset}'])
            self.featuremap_df[f'ML_prediction_1_{dataset}'] = (self.featuremap_df[f'ML_prob_1_{dataset}']>0.5).astype(int)
            self.featuremap_df[f'ML_prediction_0_{dataset}'] = (self.featuremap_df[f'ML_prob_0_{dataset}']>0.5).astype(int)
    
    def create_report(self):
        self.add_predictions_to_featuremap_df()
        # Create dataframes for test and train seperately
        pred_cols = (
            [f'ML_prob_{i}' for i in [0,1]] + 
            [f'ML_qual_{i}' for i in [0,1]] + 
            [f'ML_prediction_{i}' for i in [0,1]]
        )
        featuremap_df = {}
        for dataset, other_dataset in zip(['test', 'train'], ['train','test']):
            featuremap_df[dataset] = self.featuremap_df.copy()
            featuremap_df[dataset].drop(
                columns=[f'{col}_{other_dataset}' for col in pred_cols], 
                inplace=True
            )
            featuremap_df[dataset].rename(
                columns={f'{col}_{dataset}': col for col in pred_cols}, 
                inplace=True
            )
            # Make sure to take only reads that are indeed in the "train"/"test" sets:
            featuremap_df[dataset] = featuremap_df[dataset].loc[~featuremap_df[dataset]['ML_prob_1'].isna(), :]
        
        for (df, mrd_simulation_dataframe_file, statistics_h5_file, statistics_json_file, name,) in zip(
            [featuremap_df['test'], featuremap_df['train']],
            [
                self.test_mrd_simulation_dataframe_file,
                self.train_mrd_simulation_dataframe_file,
            ],
            [self.test_statistics_h5_file, self.train_statistics_h5_file],
            [self.test_statistics_json_file, self.train_statistics_json_file],
            ["test", "train"],
        ):
            create_report(
                models=self.classifiers,
                df=df,
                params=self.get_params(),
                report_name=name,
                out_path=self.out_path,
                base_name=self.out_basename,
                lod_filters=self.lod_filters,
                mrd_simulation_dataframe_file=mrd_simulation_dataframe_file,
                statistics_h5_file=statistics_h5_file,
                statistics_json_file=statistics_json_file,
            )

    def prepare_featuremap_for_model(self):
        """create FeatureMaps, downsampled and potentially balanced by features, to be used as train and test"""
        # prepare TP featuremaps for training
        logger.info("Preparing TP featuremaps for training and test")
        (
            self.tp_train_featuremap_vcf,
            self.tp_test_featuremap_vcf,
            self.tp_featuremap_entry_number,
            self.tp_train_set_size,
            self.tp_test_set_size,
        ) = prepare_featuremap_for_model(
            workdir=self.out_path,
            input_featuremap_vcf=self.hom_snv_featuremap,
            train_set_size=self.train_set_size
            // 2,  # half the total training size because we want equal parts TP and FP
            test_set_size=self.test_set_size // 2,  # half the total training size because we want equal parts TP and FP
            k_folds = self.k_folds, 
            pre_filter_bcftools_include=self.pre_filter,
            regions_file=self.tp_regions_bed_file,
            balanced_sampling_info_fields=self.balanced_sampling_info_fields,
            sorter_json_stats_file=self.sorter_json_stats_file,
            rng=self.rng,
            sp=self.sp,
        )
        # prepare FP featuremaps for training
        logger.info("Preparing FP featuremaps for training and test")
        (
            self.fp_train_featuremap_vcf,
            self.fp_test_featuremap_vcf,
            self.fp_featuremap_entry_number,
            self.fp_train_set_size,
            self.fp_test_set_size,
        ) = prepare_featuremap_for_model(
            workdir=self.out_path,
            input_featuremap_vcf=self.single_substitution_featuremap,
            train_set_size=self.train_set_size
            // 2,  # half the total training size because we want equal parts TP and FP
            test_set_size=self.test_set_size // 2,  # half the total training size because we want equal parts TP and FP
            k_folds = self.k_folds, 
            pre_filter_bcftools_include=self.pre_filter,
            regions_file=self.fp_regions_bed_file,
            sorter_json_stats_file=self.sorter_json_stats_file,
            rng=self.rng,
            sp=self.sp,
        )

    def create_dataframes(self):
        """create X and y dataframes for train and test"""
        # read dataframes
        df_tp_train = featuremap_to_dataframe(self.tp_train_featuremap_vcf)
        df_tp_test = featuremap_to_dataframe(self.tp_test_featuremap_vcf)
        df_fp_train = featuremap_to_dataframe(self.fp_train_featuremap_vcf)
        df_fp_test = featuremap_to_dataframe(self.fp_test_featuremap_vcf)
        # create X and y
        self.featuremap_df = pd.concat((df_tp_train, df_fp_train), ignore_index=True).astype({c: "category" for c in self.categorical_features})
        if self.use_CV:
            if self.split_folds_by_chrom:
                self.featuremap_df['fold_id'] = self.featuremap_df['chrom'].map(self.chroms_to_folds)
            else:
                #rng = np.random.default_rng(seed=14)
                N_train = self.featuremap_df.shape[0]
                self.featuremap_df['fold_id'] = self.rng.permutation(N_train) % self.k_folds 
            self.featuremap_df['label'] = (
                pd.concat(
                    [
                        pd.Series(np.ones(df_tp_train.shape[0])),
                        pd.Series(np.zeros(df_fp_train.shape[0])),
                    ],
                    ignore_index=True
                )
                .astype(bool)
                # .to_frame(name="label") # TODO: remove this line
            )
        else:
            self.featuremap_df['fold_id'] = -1 # X_train will contain all entries where self.featuremap_df['fold_id']!=0, so all of these
            # Test data should have 0 in column 'fold_id' (TODO: Check whether we might need np.nan here)
            test_featuremap_df = pd.concat((df_tp_test, df_fp_test), ignore_index=True)
            test_featuremap_df['fold_id'] = 0
            self.featuremap_df = pd.concat(
                (self.featuremap_df, test_featuremap_df), 
                ignore_index=True
            ).astype({c: "category" for c in self.categorical_features})
            self.featuremap_df['label'] = (
                pd.concat(
                    [
                        pd.Series(np.ones(df_tp_train.shape[0])),
                        pd.Series(np.zeros(df_fp_train.shape[0])),
                        pd.Series(np.ones(df_tp_test.shape[0])),
                        pd.Series(np.zeros(df_fp_test.shape[0])),
                    ], 
                    ignore_index=True
                )
                .astype(bool)
                # .to_frame(name="label") # TODO: Remove this line
            )

    def process(self):
        # prepare featuremaps
        self.prepare_featuremap_for_model()

        # load training data
        self.create_dataframes()

        # fit classifiers
        for k in range(self.k_folds):
            # datasets for k'th fold
            train_cond = np.logical_and(
                self.featuremap_df['fold_id'] != k, # Reads that are not in current test fold
                ~self.featuremap_df['fold_id'].isna() # Reads that are not in any fold.                
            )
            val_cond = (self.featuremap_df['fold_id'] == k) # Reads that are in current test fold
            X_train = self.featuremap_df.loc[train_cond, self.columns]
            y_train = self.featuremap_df.loc[train_cond, ['label']]
            X_val = self.featuremap_df.loc[val_cond, self.columns]
            y_val = self.featuremap_df.loc[val_cond, ['label']]
            # fit classifier of k'th fold
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.classifiers[k].fit(X_train, y_train, eval_set=eval_set)

        # create training and test reports
        self.create_report()

        # Calculate vector of quality scores for the test set
        quality_interpolation_function = get_quality_interpolation_function(self.test_mrd_simulation_dataframe_file)
        # probabilities = self.classifier.predict_proba(self.X_test[self.columns])[:, 0]
        self.qual_test = (
            self.featuremap_df.loc[
                ~self.featuremap_df['ML_prob_0_test'].isna(), # Make sure only "test" data is used
                'ML_prob_0_test'
            ].rename("qual").apply(quality_interpolation_function).to_frame()
        )

        # save classifier and data, generate plots for report
        self.save_model_and_data()

        return self
