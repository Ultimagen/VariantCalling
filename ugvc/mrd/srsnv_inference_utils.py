from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pysam
from pandas.api.types import CategoricalDtype

from ugvc import logger
from ugvc.vcfbed.variant_annotation import VcfAnnotator

# TODO add tests for the inference module


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
    ):
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categories = extract_categories_from_parquet(X_train_path)
        self.pre_filter = pre_filter

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
                ("ID", "ML_QUAL"),
                ("Number", 1),
                ("Type", "Float"),
                ("Description", "Single-Read-SNV model Phred score"),
            ],
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
        data = [{column: variant.info[column] for column in columns} for variant in records]
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
                predicted_probability[~df.eval(self.pre_filter.replace("INFO/", "").replace("&&", "&"))] = 1
            except Exception as e:
                logger.error(f"Failed to apply pre-filter: {self.pre_filter}")
                raise e

        # Assign the new quality value to each variant record
        for i, variant in enumerate(records):
            variant.info["ML_QUAL"] = -10 * np.log10(predicted_probability[i][0])
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


def single_read_snv_inference(
    featuremap_path: str,
    X_train_path: str,
    params_path: str,
    model_path: str,
    out_path: str,
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
    out_path : str
        Path to output featuremap
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
    )
    VcfAnnotator.process_vcf(
        annotators=[ml_qual_annotator],
        input_path=featuremap_path,
        output_path=out_path,
        process_number=process_number,
    )
