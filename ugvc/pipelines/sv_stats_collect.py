import pandas as pd
from ugbio_core import stats_utils
from ugbio_core.vcfbed import vcftools


def collect_size_type_histograms(svcall_vcf):
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
    # Read the VCF file
    vcf_df = vcftools.get_vcf_df(svcall_vcf, custom_info_fields=["SVLEN", "SVTYPE"]).query("filter=='PASS'")

    vcf_df["binned_svlens"] = pd.cut(
        vcf_df["svlen"].abs(),
        bins=[0, 100, 300, 500, 1000, 5000, 10000, 100000, 1000000, float("inf")],
        labels=["50-100", "100-300", "300-500", "0.5-1k", "1k-5k", "5k-10k", "10k-100k", "100k-1M", ">1M"],
        right=False,
    )
    type_counts = vcf_df["svtype"].value_counts()
    length_counts = vcf_df["binned_svlens"].value_counts().sort_index()

    svlens_counts_by_type = vcf_df.groupby(["svtype", "binned_svlens"]).size().unstack().fillna(0)
    svlens_counts_by_type = svlens_counts_by_type.reindex(
        columns=["50-100", "100-300", "300-500", "0.5-1k", "1k-5k", "5k-10k", "10k-100k", "100k-1M", ">1M"],
        fill_value=0,
    )
    # Only drop "CTX" if it exists
    svlens_counts_by_type = svlens_counts_by_type.drop("CTX", errors="ignore")

    return {
        "type_counts": type_counts,
        "length_counts": length_counts,
        "length_by_type_counts": svlens_counts_by_type,
    }


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
    predictions = gt["qual"].fillna(0)
    fn_mask = gt == "FN"
    gt = gt["label"]
    gt = gt.replace({"FN": "TP"})
    pos_label = "TP"
    min_class_counts_to_output = 20
    precision, recall, thresholds = stats_utils.precision_recall_curve(
        gt, predictions, fn_mask, pos_label=pos_label, min_class_counts_to_output=min_class_counts_to_output
    )
    return precision, recall, thresholds
