from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pysam

from ugvc.vcfbed.variant_annotation import VcfAnnotator


class MLQualAnnotator(VcfAnnotator):
    """
    Annotate vcf with ml-qual score
    """

    def __init__(self, model, categorical_features: list[str], numerical_features: list[str]):
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

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
                ("Description", "single-read-SNV model phred score"),
            ],
        )

        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of updated VCF records
        """
        # Convert list of variant records to a pandas DataFrame
        columns = self.numerical_features + self.categorical_features
        # TODO change this implementation to use modules from featuremap_to_dataframe - this is brittle
        data = [{column: variant.info[column] for column in columns} for variant in records]
        df = pd.DataFrame(data)[columns]

        # Mark categorical features
        df = df.astype({col: "category" for col in self.categorical_features})

        # TODO remove this patch (types should be read from header as in featuremap_to_dataframe)
        df = df.astype(
            {
                k: v
                for k, v in {
                    **{x: int for x in ("a3", "ae", "as", "s2", "s3", "te", "ts")},
                    **{"rq": float},
                }.items()
                if k in df.columns
            },
        )

        # Apply the provided model to assign a new quality value
        predicted_qualities = self.model.predict_proba(df)

        # Assign the new quality value to each variant record
        for i, variant in enumerate(records):
            variant.info["ML_QUAL"] = -10 * np.log10(predicted_qualities[i][0])
        return records


def single_read_snv_inference(
    featuremap_path: str,
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
    )
    VcfAnnotator.process_vcf(
        annotators=[ml_qual_annotator],
        input_path=featuremap_path,
        output_path=out_path,
        process_number=process_number,
    )
