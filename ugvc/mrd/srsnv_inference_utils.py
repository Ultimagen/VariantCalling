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

    def __init__(self, model, categorical_features: list[str]):
        self.model = model
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
        data = []
        features = self.model.feature_names_in_
        for variant in records:
            d_rec = {feature: variant.info[feature] for feature in features}
            data.append(d_rec)
        df = pd.DataFrame(data)[features]

        # Mark categorical features
        df = df.astype({col: "category" for col in self.categorical_features})

        # Apply the provided model to assign a new quality value
        predicted_qualities = self.model.predict_proba(df)

        # Assign the new quality value to each variant record
        for i, variant in enumerate(records):
            variant.info["ML_QUAL"] = -10 * np.log10(predicted_qualities[i][0])
        return records


def single_read_snv_inference(featuremap_path: str, params_path: str, model_path: str, out_path: str) -> None:

    # os.makedirs(os.dirname(out_path), exist_ok=True)
    model = joblib.load(model_path)
    with open(params_path, "r", encoding="UTF-8") as p_fh:
        params = json.load(p_fh)
    categorical_features = params["categorical_features"]
    ml_qual_annotator = MLQualAnnotator(model, categorical_features)
    VcfAnnotator.process_vcf(
        annotators=[ml_qual_annotator], input_path=featuremap_path, output_path=out_path, multiprocess_contigs=False
    )
