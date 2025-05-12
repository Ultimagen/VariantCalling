import pandas as pd
from ugbio_core import stats_utils
from ugbio_core.vcfbed import vcftools


def collect_size_type_histograms(svcall_vcf: str) -> dict[pd.DataFrame]:
    """
    Collect size and type histograms from SV call VCF.

    Parameters
    ----------
    svcall_vcf : str
        Path to the SV call VCF file.

    Returns
    -------
    dict[pd.DataFrame]
        Dictionary containing size and type histograms.
    """
    result = {}
    # Read the VCF file
    vcf_df = vcftools.get_vcf_df(svcall_vcf, custom_info_fields=["SVLEN", "SVTYPE"]).query("filter=='PASS'")

    type_counts = vcf_df["svtype"].value_counts()
    result["type_counts"] = type_counts
    # collect a histogram of svlens of all variants, of deletions and of insertions with power-law spaced bins
    vcf_df["svlen"] = vcf_df["svlen"].abs()
    vcf_df["svlen"].fillna(10e6, inplace=True)
    vcf_df["binned_svlens"] = pd.cut(
        vcf_df["svlen"],
        bins=[50, 100, 300, 500, 1000, 5000, 10000, 100000, 1000000, 10000000],
        labels=["50-100", "100-300", "300-500", "0.5-1k", "1k-5k", "5k-10k", "10k-100k", "100k-1M", ">1M"],
        include_lowest=False,
    )
    svlens_counts = vcf_df["binned_svlens"].value_counts().sort_index()
    result["length_counts"] = svlens_counts
    # count binned_svlens by svtype
    svlens_counts_by_type = vcf_df.groupby(["svtype", "binned_svlens"]).size().unstack().fillna(0).drop("CTX")
    result["length_by_type_counts"] = svlens_counts_by_type

    return result


def concordance_with_gt(df_base: pd.DataFrame, df_calls: pd.DataFrame) -> pd.Series:
    """
    Extract precision/recall statistics

    Parameters
    ----------
    df_base : pd.DataFrame
        DataFrame containing the base concordance.
    df_calls : pd.DataFrame
        DataFrame containing the calls concordance.

    Returns
    -------
    pd.Series
        TP,FN,FP,Precision,Recall, F1
    """
    tp_base = df_base.query("label == 'TP'").shape[0]
    tp_calls = df_calls.query("label == 'TP'").shape[0]
    fn = df_base.query("label == 'FN'").shape[0]
    fp = df_calls.query("label == 'FP'").shape[0]
    precision = tp_calls / (tp_calls + fp) if (tp_calls + fp) > 0 else 0
    recall = tp_base / (tp_base + fn) if (tp_base + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series(
        {
            "TP_base": tp_base,
            "TP_calls": tp_calls,
            "FN": fn,
            "FP": fp,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )


def concordance_with_gt_roc(df_base: pd.DataFrame, df_calls: pd.DataFrame) -> tuple:
    """
    Extract the ROC curves.

    Parameters
    ----------
    df_base : pd.DataFrame
        DataFrame containing the base concordance.
    df_calls : pd.DataFrame
        DataFrame containing the calls concordance.

    Returns
    -------
    tuple
        precision, recall, thresholds
    """
    gt = pd.concat((df_base.query("label=='FN'"), df_calls))
    predictions = gt["qual"].fillna()
    fn_mask = gt == "FN"
    gt = gt["label"]
    gt = gt.replace({"FN": "TP"})
    pos_label = "TP"
    min_class_counts_to_output = 20
    precision, recall, thresholds = stats_utils.precision_recall_curve(
        gt, predictions, fn_mask, pos_label=pos_label, min_class_counts_to_output=min_class_counts_to_output
    )
    return precision, recall, thresholds
