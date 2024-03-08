from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn import compose, impute, preprocessing
from sklearn.pipeline import make_pipeline

from ugvc.filtering.tprep_constants import VcfType


# transformers for VCF columns
def tuple_break(x):
    """Returns the first element in the tuple"""
    if isinstance(x, tuple):
        return x[0] if x[0] is not None else np.nan
    return 0 if (x is None or np.isnan(x)) else x


def tuple_break_second(x):
    """Returns the second element in the tuple"""
    if isinstance(x, tuple) and len(x) > 1:
        return x[1]
    return 0 if (x is None or (isinstance(x, tuple) and len(x) < 2) or np.isnan(x)) else x


def tuple_break_third(x):
    """Returns the third element in the tuple"""
    if isinstance(x, tuple) and len(x) > 1:
        return x[2]
    return 0 if (x is None or (isinstance(x, tuple) and len(x) < 2) or np.isnan(x)) else x


def motif_encode_left(x):
    """Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    x_list = list(x)
    x_list.reverse()
    num = 0
    for c_val in x_list:
        num = 10 * num + bases.get(c_val, 0)
    return num


def motif_encode_right(x):
    """Gets motif as input and translates it into integer
    by bases mapping and order of the bases
    The closes to the variant is the most significant bit
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
    x_list = list(x)
    num = 0
    for c_val in x_list:
        num = 10 * num + bases.get(c_val, 0)
    return num


# def trinucleotide_encode(q):
#     bases = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}

#     result = 0
#     for i in range(4):
#         result += result * 10 + bases[q[i]]
#     return result


def allele_encode(x):
    """Translate base into integer.
    In case we don't get a single base, we return zero
    """
    bases = {"A": 1, "T": 2, "G": 3, "C": 4}
    return bases.get(x, 0)


def gt_encode(x):
    """Checks whether the variant is heterozygous(0) or homozygous(1)"""
    if x == (1, 1):
        return 1
    return 0


INS_DEL_ENCODE = {"ins": -1, "del": 1, "NA": 0}


def ins_del_encode(x, encode_dct=INS_DEL_ENCODE):  # pylint: disable=dangerous-default-value
    return encode_dct[x]


def get_needed_features(vtype: VcfType = VcfType.SINGLE_SAMPLE, custom_annotations: list | None = None) -> list:
    """Get the list of features that are needed for the model

    Parameters
    ----------
    vtype: string
        The type of the input vcf. Either "single_sample", "joint" or "dv"

    custom_annotations: list, optional
        Additional custom (non-required) annotations. Some will have pre-registered custom transforms
    Returns
    -------
    list of features
    """
    features, _, _ = modify_features_based_on_vcf_type(vtype, custom_annotations)
    return features


def modify_features_based_on_vcf_type(
    vtype: VcfType = VcfType.SINGLE_SAMPLE, custom_annotations: list | None = None
) -> tuple[list, list, str]:
    """Modify training features based on the type of the vcf

    Parameters
    ----------
    vtype: string
        The type of the input vcf. Either "single_sample", "joint" or "dv"
    custom_annotations: list, optional
        Additional custom (non-required) annotations. Some will have pre-registered custom transforms
    Returns
    -------
    list of features, list of transform, column used for qual (qual or qd)

    Raises
    ------
    ValueError
        If vcf is of unrecognized type.
    """

    def tuple_encode_df(s):
        return pd.DataFrame(np.array([tuple_break(y) for y in s]).reshape((-1, 1)), index=s.index)

    def tuple_encode_doublet_df(s):
        return pd.DataFrame(np.array([y[:2] for y in s]).reshape((-1, 2)), index=s.index)

    def tuple_uniform_encode(s):
        return pd.DataFrame(list(s), index=s.index).fillna(1000)

    def motif_encode_left_df(s):
        return pd.DataFrame(np.array([motif_encode_left(y) for y in s]).reshape((-1, 1)), index=s.index)

    def motif_encode_right_df(s):
        return pd.DataFrame(np.array([motif_encode_right(y) for y in s]).reshape((-1, 1)), index=s.index)

    def allele_encode_df(s):
        return pd.DataFrame(np.array([x[:2] for x in s]), index=s.index).applymap(allele_encode)

    def allele_encode_single(df):
        return df.applymap(allele_encode)

    def gt_encode_df(s):
        return pd.DataFrame(np.array([gt_encode(y) for y in s]).reshape((-1, 1)), index=s.index)

    def ins_del_encode_df(df):
        return pd.DataFrame(np.array(df[0].apply(ins_del_encode)).reshape(-1, 1), index=df.index)

    default_filler = impute.SimpleImputer(strategy="constant", fill_value=0)
    tuple_filter = preprocessing.FunctionTransformer(tuple_encode_df)
    ins_del_encode_filter = preprocessing.FunctionTransformer(ins_del_encode_df)
    tuple_encode_df_transformer = preprocessing.FunctionTransformer(tuple_encode_df)
    tuple_encode_doublet_df_transformer = preprocessing.FunctionTransformer(tuple_encode_doublet_df)
    tuple_uniform_encode_df_transformer = preprocessing.FunctionTransformer(tuple_uniform_encode)
    left_motif_filter = preprocessing.FunctionTransformer(motif_encode_left_df)
    right_motif_filter = preprocessing.FunctionTransformer(motif_encode_right_df)
    allele_filter_single = preprocessing.FunctionTransformer(allele_encode_single)
    allele_filter = preprocessing.FunctionTransformer(allele_encode_df)
    gt_filter = preprocessing.FunctionTransformer(gt_encode_df)

    transform_list = [
        ("sor", default_filler, ["sor"]),
        ("dp", default_filler, ["dp"]),
        ("alleles", allele_filter, "alleles"),
        ("x_hin", make_pipeline(tuple_filter, allele_filter_single), "x_hin"),
        ("x_hil", make_pipeline(tuple_filter, default_filler), "x_hil"),
        ("x_il", make_pipeline(tuple_filter, default_filler), "x_il"),
        ("indel", "passthrough", ["indel"]),
        ("x_ic", make_pipeline(tuple_filter, ins_del_encode_filter), "x_ic"),
        ("x_lm", left_motif_filter, "x_lm"),
        ("x_rm", right_motif_filter, "x_rm"),
        ("x_css", make_pipeline(tuple_filter, preprocessing.OrdinalEncoder()), "x_css"),
        ("x_gcc", default_filler, ["x_gcc"]),
    ]
    features = [x[0] for x in transform_list]

    if vtype == VcfType.DEEP_VARIANT:
        transform_list.extend(
            [
                ("vaf", tuple_filter, "vaf"),
                ("ad", tuple_encode_doublet_df_transformer, "ad"),
                ("mq0_ref", default_filler, ["mq0_ref"]),
                ("mq0_alt", default_filler, ["mq0_alt"]),
                ("ls_ref", default_filler, ["ls_ref"]),
                ("ls_alt", default_filler, ["ls_alt"]),
                ("rs_ref", default_filler, ["rs_ref"]),
                ("rs_alt", default_filler, ["rs_alt"]),
                ("mean_nm_ref", default_filler, ["mean_nm_ref"]),
                ("median_nm_ref", default_filler, ["median_nm_ref"]),
                ("mean_nm_alt", default_filler, ["mean_nm_alt"]),
                ("median_nm_alt", default_filler, ["median_nm_alt"]),
                ("mean_mis_ref", default_filler, ["mean_mis_ref"]),
                ("median_mis_ref", default_filler, ["median_mis_ref"]),
                ("mean_mis_alt", default_filler, ["mean_mis_alt"]),
                ("median_mis_alt", default_filler, ["median_mis_alt"]),
                ("qual", "passthrough", ["qual"]),
            ]
        )
        features = [x[0] for x in transform_list]
    elif vtype == VcfType.SINGLE_SAMPLE:
        transform_list.extend(
            [
                ("qual", "passthrough", ["qual"]),
                ("fs", default_filler, ["fs"]),
                ("qd", default_filler, ["qd"]),
                ("mq", default_filler, ["mq"]),
                ("an", default_filler, ["an"]),
                ("baseqranksum", default_filler, ["baseqranksum"]),
                ("excesshet", default_filler, ["excesshet"]),
                ("mqranksum", default_filler, ["mqranksum"]),
                ("readposranksum", default_filler, ["readposranksum"]),
                ("ac", tuple_encode_df_transformer, "ac"),
                ("ad", tuple_encode_doublet_df_transformer, "ad"),
                ("gt", gt_filter, "gt"),
                ("xc", default_filler, ["xc"]),
                ("gq", default_filler, ["gq"]),
                ("pl", tuple_uniform_encode_df_transformer, "pl"),
                ("af", tuple_encode_df_transformer, "af"),
                ("mleac", tuple_encode_df_transformer, "mleac"),
                ("mleaf", tuple_encode_df_transformer, "mleaf"),
                ("hapcomp", tuple_encode_df_transformer, "hapcomp"),
                ("mq0c", tuple_encode_doublet_df_transformer, "mq0c"),
                ("scl", tuple_encode_doublet_df_transformer, "scl"),
                ("scr", tuple_encode_doublet_df_transformer, "scr"),
            ]
        )
        features = [x[0] for x in transform_list]
    elif vtype == VcfType.JOINT:
        pass
    else:
        raise ValueError("Unrecognized VCF type")

    # Taking care of custom annotations
    # In general we assume that custom annotations are TRUE/NONE
    # Some custom annotation require special treatment and this is defined in custom_fields_dict
    default_transformer = make_pipeline(
        impute.SimpleImputer(strategy="constant", missing_values=None, fill_value="FALSE"),
        preprocessing.OrdinalEncoder(),
    )

    def convert_to_numeric(df):
        return pd.DataFrame(pd.to_numeric(pd.Series(df.iloc[:, 0])))

    convert_to_numeric_transformer = preprocessing.FunctionTransformer(convert_to_numeric)

    long_hmer_transformer = make_pipeline(
        impute.SimpleImputer(strategy="constant", fill_value="0", missing_values=None), convert_to_numeric_transformer
    )

    if custom_annotations is None:
        custom_annotations = []
    custom_fields_dict = defaultdict(lambda: default_transformer)
    custom_fields_dict["long_hmer"] = long_hmer_transformer
    for an in custom_annotations:
        features.append(an)
        transform_list.append((an, custom_fields_dict[an], [an]))

    qual_column = "qual"
    return features, transform_list, f"{qual_column}__{qual_column}"


def get_transformer(vtype: VcfType, annots: list | None = None) -> compose.ColumnTransformer:
    """Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    vtype: VcfType
        The type of the input vcf. Either "single_sample" or "joint"
    annots: list, optional
        List of annotation features (will be transformed with "None")

    Returns
    -------
    compose.ColumnTransformer
        Transformer of the dataframe to dataframe
    """
    _, transform_list, _ = modify_features_based_on_vcf_type(vtype, annots)
    transformer = compose.ColumnTransformer(transform_list)
    transformer.set_output(transform="pandas")

    return transformer


def encode_labels(ll: Iterable[tuple[int, int]]) -> list[int]:
    """Convert genotype label (tuple) to number

    Parameters
    ----------
    ll : Iterable[tuple[int, int]]
        List of labels

    Returns
    -------
    list[int]
        List of encoded labels
    """
    return [encode_label(x) for x in ll]


def encode_label(label: tuple[int, ...]) -> int:
    "Convert genotype label (tuple) to number"
    label = tuple(sorted(label))
    if label == (0, 0):
        return 0
    if label in [(0, 1), (1, 0)]:
        return 1
    if label == (1, 1):
        return 2
    raise ValueError(f"Encoding of gt={label} not supported")


def decode_label(label: int) -> tuple[int, int]:
    "Numerical encoding of genotype to tuple of genotypes"
    decode_dct = {0: (0, 1), 1: (1, 1), 2: (0, 0)}
    return decode_dct[label]


label_encode = preprocessing.FunctionTransformer(encode_labels)
