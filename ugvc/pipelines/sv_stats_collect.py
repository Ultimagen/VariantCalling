from __future__ import annotations

import argparse
import sys

import dill as pickle
import numpy as np
import pandas as pd
from ugbio_core import stats_utils
from ugbio_core.vcfbed import vcftools

SVBINS = [0, 100, 300, 500, 1000, 5000, 10000, 100000, 1000000, float("inf")]
SVLABELS = ["50-100", "100-300", "300-500", "0.5-1k", "1k-5k", "5k-10k", "10k-100k", "100k-1M", ">1M"]


def collect_size_type_histograms(svcall_vcf, ignore_filter: bool = False) -> dict[str, pd.DataFrame]:
    """
    Collect size and type histograms from SV call VCF.

    Parameters
    ----------
    svcall_vcf: str
        Path to the SV call VCF file.
    ignore_filter: bool, optional
        Whether to ignore the FILTER field in the VCF file, by default False

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing size and type histograms.
    """
    # Read the VCF file
    if ignore_filter:
        vcf_df = vcftools.get_vcf_df(svcall_vcf, custom_info_fields=["SVLEN", "SVTYPE"])
    else:
        vcf_df = vcftools.get_vcf_df(svcall_vcf, custom_info_fields=["SVLEN", "SVTYPE"])
        vcf_df = vcf_df.query("(filter=='PASS') | (filter=='') | (filter == '.')")
    vcf_df["svlen"] = vcf_df["svlen"].apply(lambda x: x[0] if isinstance(x, tuple) else x).fillna(0)
    vcf_df["binned_svlens"] = pd.cut(
        vcf_df["svlen"].abs(),
        bins=SVBINS,
        labels=SVLABELS,
        right=False,
    )
    type_counts = vcf_df["svtype"].value_counts()
    length_counts = vcf_df["binned_svlens"].value_counts().sort_index()

    svlens_counts_by_type = vcf_df.groupby(["svtype", "binned_svlens"]).size().unstack().fillna(0)
    svlens_counts_by_type = svlens_counts_by_type.reindex(
        columns=SVLABELS,
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


def concordance_with_gt_roc(df_base: pd.DataFrame, df_calls: pd.DataFrame) -> pd.Series:
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
    pd.Series
        pandas series containing precision, recall and thresholds as values
    """
    gt = pd.concat((df_base.query("label=='FN'"), df_calls))
    predictions = gt["qual"].fillna(0)
    fn_mask = gt["label"] == "FN"
    gt = gt["label"]
    gt = gt.replace({"FN": "TP"})
    pos_label = "TP"
    MIN_CLASS_COUNTS_TO_OUTPUT = 20
    precision, recall, thresholds, _ = stats_utils.precision_recall_curve(
        np.array(gt),
        np.array(predictions),
        np.array(fn_mask),
        pos_label=pos_label,
        min_class_counts_to_output=MIN_CLASS_COUNTS_TO_OUTPUT,
    )
    return pd.Series(dict(zip(["precision", "recall", "thresholds"], [precision, recall, thresholds])))


def collect_sv_stats(
    svcall_vcf: str, concordance_h5: str | None = None, ignore_filter: bool = False
) -> tuple[dict, dict, pd.Series]:
    """
    Collect SV statistics from a VCF file. Optionally, collect also statistics about concordance with the ground truth

    Parameters
    ----------
    svcall_vcf : str
        Path to the SV call VCF file.
    concordance_h5 : str, optional
        Path to the concordance HDF5 file, by default None
    ignore_filter: bool, optional
        Whether to ignore the FILTER field in the VCF file, by default False

    Returns
    -------
    tuple[dict, dict, pd.Series]
        Dictionaries containing SV statistics with and without ground truth
        and false positive statistics seris (number of false positives
        per variant type and length)
    """
    sv_stats = collect_size_type_histograms(svcall_vcf, ignore_filter=ignore_filter)
    concordance_stats = {}
    fp_stats = pd.Series(dtype="int64")
    if concordance_h5 is not None:
        # Read the concordance HDF5 file
        df_base = pd.read_hdf(concordance_h5, key="base")
        df_calls = pd.read_hdf(concordance_h5, key="calls")
        df_base["binned_svlens"] = pd.cut(
            df_base["svlen_int"].abs(),
            bins=SVBINS,
            labels=SVLABELS,
            right=False,
        )
        df_calls["binned_svlens"] = pd.cut(
            df_calls["svlen_int"].abs(),
            bins=SVBINS,
            labels=SVLABELS,
            right=False,
        )
        # Collect concordance statistics

        category = ["ALL", "DEL", "DUP", "INV", "INS", "CTX"]
        for svtype in category:
            if svtype == "ALL":
                df_base_c = df_base
                df_calls_c = df_calls
            else:
                df_base_c = df_base.query(f"svtype == '{svtype}'")
                df_calls_c = df_calls.query(f"svtype == '{svtype}'")

            concordance_stats[f"{svtype}_concordance"] = concordance_with_gt(df_base_c, df_calls_c)  # type: ignore
            concordance_stats[f"{svtype}_roc"] = concordance_with_gt_roc(df_base_c, df_calls_c)  # type: ignore

        category = ["ALL", "DEL", "INS"]
        bins = SVLABELS
        for svtype in category:
            for len_bin in bins:
                if svtype == "ALL":
                    df_base_c = df_base
                    df_calls_c = df_calls
                else:
                    df_base_c = df_base.query(f"svtype == '{svtype}'")
                    df_calls_c = df_calls.query(f"svtype == '{svtype}'")
                df_base_c = df_base_c.query(f"binned_svlens == '{len_bin}'")
                df_calls_c = df_calls_c.query(f"binned_svlens == '{len_bin}'")
                concordance_stats[f"{svtype}_{len_bin}_concordance"] = concordance_with_gt(df_base_c, df_calls_c).drop(
                    ["FP", "Precision", "F1"]
                )
        fp_stats = (
            df_calls.query("label=='FP'")[["svtype", "binned_svlens"]].value_counts().sort_index().astype("int64")
        )

    return sv_stats, concordance_stats, fp_stats


def run(args: list[str]):
    """
    Run the SV statistics collection pipeline.

    Parameters
    ----------
    args : list[str]
        Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect SV statistics from a VCF file and (optionally) concordance H5."
        "See ugbio_comparison.sv_comparison_pipeline to get concordance between SV callsets"
    )
    parser.add_argument("svcall_vcf", type=str, help="Path to the SV call VCF file.")
    parser.add_argument("output_file", type=str, help="Output PKL file.")
    parser.add_argument(
        "--concordance_h5", type=str, help="Path to the concordance HDF5 file.", default=None, required=False
    )
    parser.add_argument(
        "--ignore_filter",
        action="store_true",
        help="Whether to ignore the FILTER field in the VCF file.",
        default=False,
    )

    p_args = parser.parse_args(args[1:])

    sv_stats, concordance_stats, fp_stats_df = collect_sv_stats(
        p_args.svcall_vcf, p_args.concordance_h5, p_args.ignore_filter
    )
    results = {}
    if concordance_stats:
        concordance_df = pd.DataFrame({k: v for k, v in concordance_stats.items() if "concordance" in k}).T
        df = pd.DataFrame(
            [x.split("_") if x.count("_") == 2 else x.replace("_", "__").split("_") for x in concordance_df.index]
        )
        df = df.drop(2, axis=1)
        df.columns = ["SV type", "SV length"]
        concordance_df = pd.concat([df, concordance_df.reset_index().drop("index", axis=1)], axis=1).set_index(
            ["SV type", "SV length"]
        )
        roc_df = pd.DataFrame({k: v for k, v in concordance_stats.items() if "roc" in k}).T
        roc_df = pd.concat([df, roc_df.reset_index().drop("index", axis=1)], axis=1).set_index(["SV type", "SV length"])
        roc_df = roc_df.rename(columns={"precision": "precision roc", "recall": "recall roc"})
        concordance_df = pd.concat((concordance_df, roc_df), axis=1)
        results["concordance"] = concordance_df
        results["fp_stats"] = fp_stats_df

    for k, v in sv_stats.items():
        results[k] = v

    with open(p_args.output_file, "wb") as f:
        pickle.dump(results, f)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
