from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
import pysam
from pandas.api.types import CategoricalDtype
from scipy.interpolate import interp1d

from ugvc import logger
from ugvc.dna.format import ALT, CHROM, FILTER, POS, QUAL, REF
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
    categorical_features : list[str]
        List of categorical features
    categorical_features : list[str] | None
        List of new categorical features (with new context variables). If None, use old context variables
    numerical_features : list[str]
        List of numerical features
    X_train_path : str
        Path to X_train
    pre_filter : str, optional
        bcftools expression used to filter variants before training the model, as part of:
        "bcftools view <vcf> -i 'pre_filter'"
        Any variant that does not pass the filter will be assigned ML_QUAL=0 in inference.
        Default None
    """

    def __init__(
        self,
        models,
        categorical_features: list[str],
        numerical_features: list[str],
        X_train_path: str,
        pre_filter: str = None,
        test_set_mrd_simulation_dataframe_file: str = None,
        low_qual_threshold: float = LOW_QUAL_THRESHOLD,
        new_categorical_features: list[str] = None,
        chrom_folds: pd.DataFrame = None,
    ):
        self.models = models
        self.num_folds = len(models)
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.new_categorical_features = new_categorical_features
        self.categories = extract_categories_from_parquet(
            X_train_path
        )  # TODO: Need to make sure that I'm extracting the new categorical features
        if pre_filter:
            self.pre_filter = pre_filter.replace("INFO/", "").replace("&&", "&").replace("=", "==")
            logger.debug(f"Using pre-filter query: {self.pre_filter} (original: {pre_filter})")
        else:
            self.pre_filter = None
        self.quality_interpolation_function = get_quality_interpolation_function(test_set_mrd_simulation_dataframe_file)
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

    def add_new_context_columns(self, df):
        for i in range(3):
            df[f"next_{i+1}"] = df["next_3bp"].str[i]
        for i in range(3):
            df[f"prev_{i+1}"] = df["prev_3bp"].str[-(i + 1)]
        return df

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
        columns = self.numerical_features + self.categorical_features + ["ref", "alt", "chrom"]
        data = [{column: _vcf_getter(variant, column) for column in columns} for variant in records]
        df = pd.DataFrame(data)[columns]
        if self.new_categorical_features is not None:
            df = self.add_new_context_columns(df)

        # encode categorical features
        for column, cat_values in self.categories.items():
            if column in df.columns:
                cat_dtype = CategoricalDtype(categories=cat_values, ordered=False)
                df[column] = df[column].astype(cat_dtype)
                df[column] = df[column].cat.set_categories(new_categories=cat_values, ordered=False)
        if self.new_categorical_features is not None:
            for col in ("alt", "ref", "next_1", "next_2", "next_3", "prev_1", "prev_2", "prev_3"):
                if col in df.columns:
                    df[col] = df[col].astype(
                        CategoricalDtype(categories=["A", "C", "G", "T"], ordered=False)
                    )  # Categories of new features are always A, C, G, T
                    df[col] = df[col].cat.set_categories(new_categories=["A", "C", "G", "T"], ordered=False)

        # df['X_SMQ_LEFT_MEAN'] = df['X_SMQ_LEFT']
        # df['X_SMQ_RIGHT_MEAN'] = df['X_SMQ_RIGHT']

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
        features_for_model = (
            self.numerical_features + self.categorical_features
            if self.new_categorical_features is None
            else self.numerical_features + self.new_categorical_features
        )
        if self.num_folds == 1:
            # No CV
            predicted_probability = self.models[0].predict_proba(df[features_for_model])
        else:
            predicted_probability = np.zeros((df.shape[0], 2))
            df["fold"] = df["chrom"].map(self.chrom_folds["fold"].to_dict(), na_action="ignore")
            for k, model in enumerate(self.models):
                if (df["fold"] == k).any():
                    predicted_probability[(df["fold"] == k).index, :] = model.predict_proba(
                        df.loc[df["fold"] == k, features_for_model]
                    )  # predict on chromosomes of fold k
                if df["fold"].isna().any():
                    predicted_probability[df["fold"].isna().index, :] += (
                        model.predict_proba(df.loc[df["fold"].isna(), features_for_model]) / self.num_folds
                    )  # chromosomes not in any fold, predict with all models and take mean

        if self.pre_filter:
            try:
                # the bcftools filter should work as a query if applied on info fields because their names are the same
                # in the vcf header and in the dataframe. However, the bcftools expression could explicitly contain
                # INFO/, e.g INFO/X_SCORE>4, so we need to remove the INFO/ prefix before applying the filter
                # We also replace && with & because the former is supported by bcftools and the latter by pandas
                # We assign a value of 1 for error probability, translating to a quality of 0
                predicted_probability[~df.eval(self.pre_filter)] = 1
            except Exception as e:
                logger.error(f"Failed to apply pre-filter: {self.pre_filter}")
                raise e

        # Assign the new quality value to each variant record
        for i, variant in enumerate(records):
            ml_qual = (-10 * np.log10(predicted_probability[i][0])).clip(
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


def get_quality_interpolation_function(mrd_simulation_dataframe: str = None):
    """
    Create a function that interpolates between residual_snv_rate and ML_QUAL

    Parameters
    ----------
    mrd_simulation_dataframe : str, None
        Path to MRD simulation dataframe
        If None then None is returned and nothing is done

    Returns
    -------
    interp1d
        Interpolation function
    """
    if not mrd_simulation_dataframe:
        return None
    # read data, filter for ML_QUAL filters only
    df = pd.read_parquet(mrd_simulation_dataframe)
    df.index = df.index.str.upper()
    df = df[df.index.str.startswith(ML_QUAL)]
    df.loc[:, ML_QUAL] = df.index.str.replace(ML_QUAL + "_", "").astype(float)
    # Calculate Phred scores matching the residual SNV rate
    df = df[df["residual_snv_rate"] > 0]  # Estimate using only residual SNV rates > 0 to avoid QUAL=Inf
    phred_residual_snv_rate = -10 * np.log10(df["residual_snv_rate"])
    # Create interpolation function
    quality_interpolation_function = interp1d(
        df[ML_QUAL] + 1e-10,  # add a small number so ML_QUAL=0 is outside the interpolation range
        phred_residual_snv_rate,
        kind="linear",
        bounds_error=False,
        fill_value=(
            0,
            phred_residual_snv_rate[-1],
        ),  # below ML_QUAL=1E-10 assign 0, above max assign max
    )
    return quality_interpolation_function


def single_read_snv_inference(
    featuremap_path: str,
    X_train_path: str,
    params_path: str,
    model_path: str,
    test_set_mrd_simulation_dataframe_file: str,
    out_path: str,
    low_qual_threshold: float = LOW_QUAL_THRESHOLD,
    process_number: int = 0,
    num_folds: int = 0,
    chrom_folds_path: str = None,
) -> None:
    """
    Annotate a featuremap with single-read-SNV model ML_QUAL scores

    Parameters
    ----------
    featuremap_path : str
        Path to featuremap
    X_train_path : str
        Path to X_train
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
    num_folds: int, optional
        Number of cross-validation folds.  If 0, do not use CV. Default 0
    chrom_folds_path:
        The path to a csv file with information about which chromosomes belong to which CV fold.
        By default, use the following grouping into folds:
        {'chr2': 0, 'chr6': 0, 'chr22': 0, 'chr14': 0,
        'chr3': 1, 'chr4': 1, 'chr5': 1,
        'chr7': 2, 'chr8': 2, 'chr9': 2, 'chr11': 2,
        'chr1': 3, 'chr20': 3, 'chr10': 3, 'chr12': 3,
        'chr15': 4, 'chr16': 4, 'chr17': 4, 'chr18': 4, 'chr19': 4, 'chr13': 4, 'chr21': 4}
    """
    # Load model / models for cv
    if num_folds == 0:
        models = [joblib.load(model_path)]
        logger.info("Cross-validation off, using single model for inference. Model loaded")
    else:
        base_name, extension = model_path.rsplit(".", 1)
        if base_name.rsplit("_", 2)[1] == "fold":
            # If model filename is like "XXXXX_fold_0.joblib":
            model_fnames = [".".join([base_name[:-1] + f"{k}", extension]) for k in range(num_folds)]
        else:
            # If model filename is like "XXXXX.joblib":
            model_fnames = [".".join([base_name + f"_fold_{k}", extension]) for k in range(num_folds)]
        models = [joblib.load(model_fname) for model_fname in model_fnames]
        logger.info(f"Cross-validation on, number of folds: {num_folds}. Models loaded")
        if chrom_folds_path is None:
            chrom_folds = pd.Series(
                {
                    "chr2": 0,
                    "chr6": 0,
                    "chr22": 0,
                    "chr14": 0,
                    "chr3": 1,
                    "chr4": 1,
                    "chr5": 1,
                    "chr7": 2,
                    "chr8": 2,
                    "chr9": 2,
                    "chr11": 2,
                    "chr1": 3,
                    "chr20": 3,
                    "chr10": 3,
                    "chr12": 3,
                    "chr15": 4,
                    "chr16": 4,
                    "chr17": 4,
                    "chr18": 4,
                    "chr19": 4,
                    "chr13": 4,
                    "chr21": 4,
                }
            ).to_frame(name="fold")
        else:
            chrom_folds = pd.read_csv(chrom_folds_path)

    with open(params_path, "r", encoding="UTF-8") as p_fh:
        params = json.load(p_fh)
    ml_qual_annotator = MLQualAnnotator(
        models,
        categorical_features=params["categorical_features"],
        new_categorical_features=params.get("new_categorical_features", None),
        numerical_features=params["numerical_features"],
        X_train_path=X_train_path,
        pre_filter=params["pre_filter"],
        test_set_mrd_simulation_dataframe_file=test_set_mrd_simulation_dataframe_file,
        low_qual_threshold=low_qual_threshold,
        chrom_folds=chrom_folds,
    )
    VcfAnnotator.process_vcf(
        annotators=[ml_qual_annotator],
        input_path=featuremap_path,
        output_path=out_path,
        process_number=process_number,
    )
