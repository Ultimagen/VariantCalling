from __future__ import annotations

from collections import OrderedDict
from enum import Enum

import numpy as np
import pandas as pd
import xgboost
from sklearn import compose
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from ugvc import logger
from ugvc.filtering import transformers
from ugvc.filtering.tprep_constants import GtType, VcfType
from ugvc.utils import math_utils


def train_model(
    concordance: pd.DataFrame,
    gt_type: GtType,
    vtype: VcfType,
    annots: list | None = None,
) -> tuple[compose.ColumnTransformer, xgboost.XGBRFClassifier]:
    """Trains model xgboost model on a subset of dataframe

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    gt_type: GtType
        Is the ground truth approximate or exact (in the first case the model will predict 0/1, in the second: 0/1/2)
    vtype: string
        The type of the input vcf. Either "single_sample" or "joint"
    annots: list, optional
        Optional list of annotations present in the dataframe

    Returns
    -------
    tuple:
        Trained transformer and classifier model

    Raises
    ------
    ValueError
        If the gt_type is not recognized
    """
    transformer = transformers.get_transformer(vtype, output_df=True, annots=annots)
    if gt_type == GtType.APPROXIMATE:
        select_train = concordance["label"].apply(lambda x: x in {0, 1})
    elif gt_type == GtType.EXACT:
        select_train = concordance["label"].apply(lambda x: x in {(0, 0), (0, 1), (1, 1), (1, 0)})
    else:
        raise ValueError("Unknown gt_type")
    df_train = concordance.loc[select_train]
    if gt_type == GtType.APPROXIMATE:
        labels_train = df_train["label"].values
    elif gt_type == GtType.EXACT:
        labels_train = transformers.label_encode.transform(list(df_train["label"].values))
    else:
        raise ValueError("Unknown gt_type")

    x_train_df = pd.DataFrame(transformer.fit_transform(df_train))
    _validate_data(x_train_df)
    clf = xgboost.XGBClassifier(
        n_estimators=100,
        learning_rate=0.15,
        subsample=0.4,
        max_depth=6,
        random_state=0,
        colsample_bytree=0.4,
        n_jobs=14,
    )
    clf.fit(x_train_df, labels_train)
    return clf, transformer


def apply_model(
    input_df: pd.DataFrame, model: xgboost.XGBClassifier, transformer: compose.ColumnTransformer
) -> tuple[np.ndarray, np.ndarray]:
    """Applies model to the input dataframe

    Parameters
    ----------
    input_df: pd.DataFrame
        Input dataframe
    model: xgboost.XGBClassifier
        Model
    transformer: compose.ColumnTransformer
        Transformer

    Returns
    -------
    tuple:
        Predictions and probabilities
    """
    x_test_df = transformer.transform(input_df)
    _validate_data(pd.DataFrame(x_test_df))
    predictions = model.predict(x_test_df)
    probabilities = model.predict_proba(x_test_df)
    return predictions, probabilities


def _validate_data(data: np.ndarray | pd.Series | pd.DataFrame) -> None:
    """Validates that the data does not contain nulls"""

    if isinstance(data, np.ndarray):
        test_data = data
    else:
        test_data = pd.DataFrame(data).to_numpy()
    try:
        if len(test_data.shape) == 1 or test_data.shape[1] <= 1:
            assert pd.isnull(test_data).sum() == 0, "data vector contains null"
        else:
            for c_val in range(test_data.shape[1]):
                assert pd.isnull(test_data[:, c_val]).sum() == 0, f"Data matrix contains null in column {c_val}"
    except AssertionError as af_val:
        logger.error(str(af_val))
        raise af_val


def add_grouping_column(df: pd.DataFrame, selection_functions: dict, column_name: str) -> pd.DataFrame:
    """
    Add a column for grouping according to the values of selection functions

    Parameters
    ----------
    df: pd.DataFrame
        concordance dataframe
    selection_functions: dict
        Dictionary of selection functions to be applied on the df, keys - are the name of the group
    column_name: str
        Name of the column to contain grouping

    Returns
    -------
    pd.DataFrame
        df with column_name added to it that is filled with the group name according
        to the selection function
    """
    df[column_name] = None
    for k in selection_functions:
        df.loc[selection_functions[k](df), column_name] = k
    return df


def get_testing_selection_functions() -> OrderedDict:
    """Return dictionary of categories functions"""
    sfs = OrderedDict()
    sfs["SNP"] = lambda x: np.logical_not(x.indel)
    hil = np.array(lambda x: x["x_hil"].apply(lambda x: x[0])).astype(int)
    sfs["Non-hmer INDEL"] = lambda x: x.indel & (hil == 0)
    sfs["HMER indel <= 4"] = lambda x: x.indel & (hil > 0) & (hil < 5)
    sfs["HMER indel (4,8)"] = lambda x: x.indel & (hil >= 5) & (hil < 8)
    sfs["HMER indel [8,10]"] = lambda x: x.indel & (hil >= 8) & (hil <= 10)
    sfs["HMER indel 11,12"] = lambda x: x.indel & (hil >= 11) & (hil <= 12)
    sfs["HMER indel > 12"] = lambda x: x.indel & (hil > 12)
    return sfs


def eval_model(
    df: pd.DataFrame,
    model: xgboost.XGBClassifier,
    transformer: compose.ColumnTransformer,
    add_testing_group_column: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate precision/recall for the decision tree classifier

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe
    model: xgboost.XGBClassifier
        Model
    transformer: compose.ColumnTransformer
        Data prep transformer
    add_testing_group_column: bool
        Should default testing grouping be added (default: True),
        if False will look for grouping in `group_testing`

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]:
        recall/precision for each category and recall precision curve for each category in dataframe

    Raises
    ------
    RuntimeError
        If the group_testing column is not present in the dataframe
    """
    df = df.copy()
    probs, predictions = apply_model(df, model, transformer)
    phred_pls = math_utils.phred(probs)
    sorted_pls = np.sort(phred_pls, axis=1)
    gqs = sorted_pls[:, -1] - sorted_pls[:, -2]
    quals = phred_pls[:, 1:].max(axis=1) - phred_pls[:, 0]
    df["ml_gq"] = gqs
    df["ml_qual"] = quals
    df["predict"] = predictions

    if add_testing_group_column:
        df = add_grouping_column(df, get_testing_selection_functions(), "group_testing")
        groups = list(get_testing_selection_functions().keys())

    else:
        assert "group_testing" in df.columns, "group_testing column should be given"
        groups = list(set(df["group_testing"]))

    labels = df["label"]
    if probs.shape[1] == 2:
        gt_type = GtType.APPROXIMATE
    elif probs.shape[1] == 3:
        gt_type = GtType.EXACT
    else:
        raise RuntimeError("Unknown gt_type")

    if gt_type == GtType.APPROXIMATE:
        select = df["label"].apply(lambda x: x in {0, 1})
    elif gt_type == GtType.EXACT:
        select = df["label"].apply(lambda x: x in {(0, 0), (0, 1), (1, 1), (1, 0)})
    labels = labels[select]
    if gt_type == GtType.EXACT:
        labels = transformers.label_encode.transform(list(labels))
    df = df.loc[select]

    accuracy_df = pd.DataFrame(
        columns=[
            "group",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "gt_precision",
            "gt_recall",
            "f1",
        ]
    )
    curve_df = pd.DataFrame(columns=["group", "precision", "recall", "f1", "threshold"])
    for g_val in groups:
        select = df["group_testing"] == g_val
        group_df = df[select]
        group_labels = labels[select]
        tp = (group_df["predict"] > 0) & (group_labels > 0)
        fp = (group_df["predict"] > 0) & (group_labels == 0)
        fn = (group_df["predict"] == 0) & (group_labels > 0)

        precision = precision_score(group_labels > 0, group_df["predict"] > 0)
        recall = recall_score(group_labels > 0, group_df["predict"] > 0)
        f1 = f1_score(group_labels > 0, group_df["predict"] > 0)
        prc = precision_recall_curve(group_labels > 0, group_df["ml_qual"])
        gt_select = (group_labels > 0) & (group_df["predict"] > 0)
        gt_precision = precision_score(group_labels[gt_select], group_df["predict"][gt_select], pos_label=2)
        gt_recall = recall_score(group_labels[gt_select], group_df["predict"][gt_select], pos_label=2)

        accuracy_df = pd.concat(
            (
                accuracy_df,
                pd.DataFrame(
                    {
                        "group": g_val,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "gt_precision": gt_precision,
                        "gt_recall": gt_recall,
                    },
                    index=[0],
                ),
            ),
            ignore_index=True,
        )
        curve_df = pd.concat(
            (
                curve_df,
                pd.DataFrame(
                    {
                        "group": g_val,
                        "precision": prc[0],
                        "recall": prc[1],
                        "f1": 2 * prc[0] * prc[1] / (prc[0] + prc[1]),
                        "threshold": prc[2],
                    },
                    index=[0],
                ),
            ),
            ignore_index=True,
        )

    return accuracy_df, curve_df


def get_empty_recall_precision(category: str) -> dict:
    """Return empty recall precision dictionary for category given"""
    return {
        "group": category,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "gt_precision": 1.0,
        "gt_recall": 1.0,
    }


def get_empty_recall_precision_curve(category: str) -> dict:
    """Return empty recall precision curve dictionary for category given"""
    return {"group": category, "precision": [], "recall": [], "f1": [], "threshold": 0}


class VariantSelectionFunctions(Enum):
    """Collecton of variant selection functions - all get DF as input and return boolean np.array"""

    @staticmethod
    def ALL(df: pd.DataFrame) -> np.ndarray:
        return np.ones(df.shape[0], dtype=bool)

    @staticmethod
    def HMER_INDEL(df: pd.DataFrame) -> np.ndarray:
        return np.array(df.hmer_indel_length > 0)

    @staticmethod
    def ALL_except_HMER_INDEL_greater_than_or_equal_5(df: pd.DataFrame) -> np.ndarray:
        return np.array(~((df.hmer_indel_length >= 5)))
