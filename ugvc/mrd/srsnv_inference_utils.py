from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pysam
from pandas.api.types import CategoricalDtype
from scipy.interpolate import interp1d

from ugvc import logger
from ugvc.vcfbed.variant_annotation import VcfAnnotator

# TODO add tests for the inference module

ML_QUAL = "ML_QUAL"
PASS = "PASS"
LOW_QUAL = "LowQual"
PRE_FILTERED = "PreFiltered"

LOW_QUAL_THRESHOLD = 40


class MLQualAnnotator(VcfAnnotator):
    """
    Annotate vcf with ml-qual score

    Parameters
    ----------
    model : sklearn.BaseEstimator
        Trained model
    categorical_features : list[str]
        List of categorical features
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
        model,
        categorical_features: list[str],
        numerical_features: list[str],
        X_train_path: str,
        pre_filter: str = None,
        test_set_mrd_simulation_dataframe_file: str = None,
        low_qual_threshold: float = LOW_QUAL_THRESHOLD,
    ):
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categories = extract_categories_from_parquet(X_train_path)
        if pre_filter:
            self.pre_filter = pre_filter.replace("INFO/", "").replace("&&", "&")
            logger.debug(f"Using pre-filter query: {self.pre_filter} (original: {pre_filter})")
        else:
            self.pre_filter = None
        self.quality_interpolation_function = get_quality_interpolation_function(test_set_mrd_simulation_dataframe_file)
        self.low_qual_threshold = low_qual_threshold

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
            header.filters.add(
                id=PRE_FILTERED,
                number=None,
                type=None,
                description=f"Variant failed SRSNV pre-filter: {self.pre_filter}",
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
        columns = self.numerical_features + self.categorical_features
        data = [
            {
                **{column: variant.info[column] for column in columns if column not in ["ref", "alt"]},
                **{"ref": variant.ref, "alt": variant.alts[0]},
            }
            for variant in records
        ]
        df = pd.DataFrame(data)[columns]

        # encode categorical features
        for column, cat_values in self.categories.items():
            if column in df.columns:
                cat_dtype = CategoricalDtype(categories=cat_values, ordered=False)
                df[column] = df[column].astype(cat_dtype)

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
        predicted_probability = self.model.predict_proba(df)
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
    phread_residual_snv_rate = -10 * np.log10(df["residual_snv_rate"])
    # Create interpolation function
    quality_interpolation_function = interp1d(
        df[ML_QUAL] + 1e-10,  # add a small number so ML_QUAL=0 is outside the interpolation range
        phread_residual_snv_rate,
        kind="linear",
        bounds_error=False,
        fill_value=(
            0,
            phread_residual_snv_rate[-1],
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

    """
    model = joblib.load(model_path)
    with open(params_path, "r", encoding="UTF-8") as p_fh:
        params = json.load(p_fh)
    ml_qual_annotator = MLQualAnnotator(
        model,
        categorical_features=params["categorical_features"],
        numerical_features=params["numerical_features"],
        X_train_path=X_train_path,
        pre_filter=params["pre_filter"],
        test_set_mrd_simulation_dataframe_file=test_set_mrd_simulation_dataframe_file,
        low_qual_threshold=low_qual_threshold,
    )
    VcfAnnotator.process_vcf(
        annotators=[ml_qual_annotator],
        input_path=featuremap_path,
        output_path=out_path,
        process_number=process_number,
    )
