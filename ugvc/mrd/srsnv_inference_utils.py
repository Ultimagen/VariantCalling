from __future__ import annotations

import json
import re
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd
import pysam
import xgboost as xgb

from ugvc import logger
from ugvc.dna.format import ALT, CHROM, FILTER, POS, QUAL, REF
from ugvc.mrd.srsnv_training_utils import (
    FOLD_ID,
    get_quality_interpolation_function,
    k_fold_predict_proba,
    set_categorical_columns,
)
from ugvc.vcfbed.variant_annotation import VcfAnnotator

# TODO add tests for the inference module

ML_QUAL = "ML_QUAL"
PASS = "PASS"
LOW_QUAL = "LowQual"
PRE_FILTERED = "PreFiltered"

LOW_QUAL_THRESHOLD = 40


def _vcf_getter(variant, field):
    if field == ALT:
        return variant.alts[0]
    if field == REF:
        return variant.ref
    if field == CHROM:
        return variant.chrom
    if field == POS:
        return variant.pos
    if field == FILTER:
        return variant.filter
    if field == QUAL:
        return variant.qual
    return variant.info.get(field, None)


def sanitize_for_vcf_general(input_string):
    # Allow only a specific set of characters, escape others
    # This regex will match any character that is NOT a safe character
    unsafe_chars = re.compile(r"[^a-zA-Z0-9 _.,;:|\-<>=()&]")

    # Replace unsafe characters with an underscore or some other placeholder
    # Or you could choose to remove them entirely by replacing with an empty string
    sanitized = unsafe_chars.sub("", input_string)

    return sanitized


class MLQualAnnotator(VcfAnnotator):
    """
    Annotate vcf with ml-qual score

    Parameters
    ----------
    models : list[sklearn.BaseEstimator]
        Trained models. If len(models)==1, do not use cross validation
    categorical_features : dict[str, list]
        Dict whose keys are categorical features and values are lists of the corresponding categories
    numerical_features : list[str]
        List of numerical features
    pre_filter : str, optional
        bcftools expression used to filter variants before training the model, as part of:
        "bcftools view <vcf> -i 'pre_filter'"
        Any variant that does not pass the filter will be assigned ML_QUAL=0 in inference.
        Default None
    quality_interpolation_function: Callable
        The function used to map ML_qual values to SNVQ values
    chrom_folds: dict[str, int]
        A dict mapping contig names (chromosomes) to the corresponding fold
    """

    def __init__(
        self,
        models,
        categorical_features_dict: dict[str, list],
        numerical_features: list[str],
        quality_interpolation_function: Callable,
        pre_filter: str = None,
        low_qual_threshold: float = LOW_QUAL_THRESHOLD,
        chrom_folds: dict = None,
    ):
        self.models = models
        self.num_folds = len(models)
        self.numerical_features = numerical_features
        self.categorical_features_dict = categorical_features_dict
        self.categorical_features_names = list(self.categorical_features_dict.keys())
        if pre_filter:
            self.pre_filter = pre_filter.replace("INFO/", "").replace("&&", "&")
            logger.debug(f"Using pre-filter query: {self.pre_filter} (original: {pre_filter})")
        else:
            self.pre_filter = None
        self.quality_interpolation_function = quality_interpolation_function
        self.low_qual_threshold = low_qual_threshold
        self.chrom_folds = chrom_folds

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to include new fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            Modified VCF header

        """
        header.add_meta(
            "INFO",
            items=[
                ("ID", ML_QUAL),
                ("Number", 1),
                ("Type", "Float"),
                ("Description", "Single-Read-SNV model Phred score"),
            ],
        )
        header.filters.add(
            id=LOW_QUAL,
            number=None,
            type=None,
            description=f"SNV quality is below {self.low_qual_threshold}",
        )
        if self.pre_filter:
            sanitized_pre_filter = sanitize_for_vcf_general(self.pre_filter)
            header.filters.add(
                id=PRE_FILTERED,
                number=None,
                type=None,
                description=f"Variant failed SRSNV pre-filter: {sanitized_pre_filter}",
            )

        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """
        Apply trained model and annotate a list of VCF records with ML_QUAL scores

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of updated VCF records

        Raises
        ------
        Exception
            If the pre-filter is not valid
        """
        # Convert list of variant records to a pandas DataFrame
        columns = self.numerical_features + self.categorical_features_names + ["chrom"]
        data = [{column: _vcf_getter(variant, column) for column in columns} for variant in records]
        df = pd.DataFrame(data)[columns]
        df = set_categorical_columns(df, self.categorical_features_dict)  # for correct encoding of categorical features

        # TODO remove this patch (types should be read from header as in featuremap_to_dataframe)
        df = df.astype(
            {
                k: v
                for k, v in {
                    **{"rq": float},
                }.items()
                if k in df.columns
            },
        )

        # Apply the provided model to assign a new quality value
        features_for_model = self.numerical_features + self.categorical_features_names
        if self.num_folds == 1:
            df[FOLD_ID] = 0
        elif self.chrom_folds is None:
            df[FOLD_ID] = np.nan  # For the case where fold-splitting was done randomly
        else:
            df[FOLD_ID] = df["chrom"].map(self.chrom_folds, na_action="ignore")
        predicted_probability = k_fold_predict_proba(
            self.models, df, features_for_model, self.num_folds, kfold_col=FOLD_ID
        ).values

        if self.pre_filter:
            try:
                # the bcftools filter should work as a query if applied on info fields because their names are the same
                # in the vcf header and in the dataframe. However, the bcftools expression could explicitly contain
                # INFO/, e.g INFO/X_SCORE>4, so we need to remove the INFO/ prefix before applying the filter
                # We also replace && with & because the former is supported by bcftools and the latter by pandas
                # We assign a value of 0 for true probability, meaning 1 for error probability, translating to a
                # quality of 0
                predicted_probability[~df.eval(self.pre_filter)] = 0
            except Exception as e:
                logger.error(f"Failed to apply pre-filter: {self.pre_filter}")
                raise e

        # Assign the new quality value to each variant record
        for i, variant in enumerate(records):
            ml_qual = (-10 * np.log10(1 - predicted_probability[i])).clip(
                min=0
            )  # clipping to avoid rounding errors (-0)
            variant.info[ML_QUAL] = ml_qual
            if self.quality_interpolation_function:
                # assign quality based on the residual SNV rate
                qual = np.around(self.quality_interpolation_function(ml_qual), decimals=2)
                variant.qual = qual

                # Add a filter
                if qual >= self.low_qual_threshold:
                    variant.filter.add(PASS)
                elif qual > 0 or not self.pre_filter:
                    variant.filter.add(LOW_QUAL)
                else:
                    variant.filter.add(PRE_FILTERED)
        return records


def extract_categories_from_parquet(parquet_path):
    """
    Extract explicit categories of categorical columns from a parquet file

    Parameters
    ----------
    parquet_path : str
        Path to parquet file

    Returns
    -------
    dict
        Dictionary of column names and categories

    """
    df = pd.read_parquet(parquet_path)
    category_mappings = {}
    for column in df.select_dtypes(include=["category"]).columns:
        category_mappings[column] = df[column].cat.categories.tolist()
    return category_mappings


def load_models(model_path: str, num_folds: int, model_params: dict):
    """Load model(s) from model_path"""
    base_name, extension = model_path.rsplit(".", 1)
    if extension == "joblib":
        load_joblib = True
    elif extension == "json":
        load_joblib = False
    else:
        raise ValueError(f"model_path must be either a joblib file or a json file, got {model_path}.")
    if num_folds == 1:
        if load_joblib:
            models = [joblib.load(model_path)]
        else:
            clf = xgb.XGBClassifier(**model_params)
            clf.load_model(model_path)
            models = [clf]
        logger.info("Cross-validation off, using single model for inference. Model loaded")
    else:
        if base_name.rsplit("_", 2)[1] == "fold":
            # If model filename is like "XXXXX_fold_0.joblib":
            model_fnames = [".".join([base_name[:-1] + f"{k}", extension]) for k in range(num_folds)]
        else:
            # If model filename is like "XXXXX.joblib":
            model_fnames = [".".join([base_name + f"_fold_{k}", extension]) for k in range(num_folds)]
        if load_joblib:
            models = [joblib.load(model_fname) for model_fname in model_fnames]
        else:
            models = []
            for fname in model_fnames:
                clf = xgb.XGBClassifier(**model_params)
                clf.load_model(fname)
                models.append(clf)
        logger.info(f"Cross-validation on, number of folds: {num_folds}. Models loaded")
    return models


def single_read_snv_inference(
    featuremap_path: str,
    model_joblib_path: str,
    params_path: str,
    model_path: str,
    test_set_mrd_simulation_dataframe_file: str,
    out_path: str,
    low_qual_threshold: float = LOW_QUAL_THRESHOLD,
    process_number: int = 0,
) -> None:
    """
    Annotate a featuremap with single-read-SNV model ML_QUAL scores

    Parameters
    ----------
    featuremap_path : str
        Path to featuremap
    model_joblib_path : str
        Path to joblib file containing information about the model, params, and interpolating function.
    params_path : str
        Path to model parameters
    model_path : str
        Path to model
    test_set_mrd_simulation_dataframe_file : str
        Path to MRD simulation dataframe
    out_path : str
        Path to output featuremap
    low_qual_threshold : float
        Threshold for low quality variants, default 40
    process_number: int, optional
        Number of processes to use for parallelization. If N < 1, use all-available - abs(N) cores. Default 0
    """
    if model_joblib_path is not None:
        model_info_dict = joblib.load(model_joblib_path)
        models = model_info_dict["models"]
        params = model_info_dict["params"]
        quality_interpolation_function = model_info_dict["quality_interpolation_function"]
        using_jl = True
    else:
        assert (model_path is not None) and (
            params_path is not None
        ), "When --model_joblib_path is not provided, must provide model_path and params_path"
        using_jl = False
    if params_path is not None:
        with open(params_path, "r", encoding="UTF-8") as p_fh:
            params = json.load(p_fh)
        if using_jl:
            logger.info("Both model_jl_path and params_path provided, Loading params from the latter")
    num_folds = params["num_CV_folds"]
    if model_path is not None:
        models = load_models(model_path, num_folds, params["model_params"])
        if using_jl:
            logger.info("Both model_jl_path and model_path provided, Loading model(s) from the latter")
    if (test_set_mrd_simulation_dataframe_file is not None) or not using_jl:
        quality_interpolation_function = get_quality_interpolation_function(test_set_mrd_simulation_dataframe_file)
        if using_jl:
            logger.info("Both model_jl_path and test_set_mrd_simulation_dataframe_file provided, using latter")
    if num_folds > 1:
        chrom_folds = params["chroms_to_folds"]
    else:
        chrom_folds = None
    assert num_folds == len(models), f"num_folds should equal number of models. Got {num_folds=}, {len(models)=}"

    ml_qual_annotator = MLQualAnnotator(
        models,
        categorical_features_dict=params["categorical_features_dict"],
        numerical_features=params["numerical_features"],
        pre_filter=params["pre_filter"],
        quality_interpolation_function=quality_interpolation_function,
        low_qual_threshold=low_qual_threshold,
        chrom_folds=chrom_folds,
    )
    VcfAnnotator.process_vcf(
        annotators=[ml_qual_annotator],
        input_path=featuremap_path,
        output_path=out_path,
        process_number=process_number,
    )
