import json
import os
from test import get_resource_dir

import numpy as np
import pandas as pd

from ugvc.methylation import (
    concat_methyldackel_csvs,
    process_Mbias,
    process_mergeContext,
    process_mergeContextNoCpG,
    process_perRead,
)
from ugvc.methylation.methyldackel_utils import (
    calc_coverage_methylation,
    calc_percent_methylation,
    calc_TotalCpGs,
    get_dict_from_dataframe,
)


class TestParsers:
    inputs_dir = get_resource_dir(__file__)

    def test_process_mbias(self, tmpdir):
        output_prefix = f"{tmpdir}/output_Mbias"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_Mbias.run(
            [
                "process_Mbias",
                "--input",
                f"{self.inputs_dir}/input_Mbias.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        ref_csv = pd.read_csv(open(f"{self.inputs_dir}/ProcessMethylDackelMbias.csv"))

        pd.testing.assert_frame_equal(result_csv, ref_csv)

    # ------------------------------------------------------

    def test_process_perRead(self, tmpdir):
        output_prefix = f"{tmpdir}/output_perRead"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_perRead.run(
            [
                "process_perRead",
                "--input",
                f"{self.inputs_dir}/input_perRead.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        ref_csv = pd.read_csv(open(f"{self.inputs_dir}/ProcessMethylDackelPerRead.csv"))

        pd.testing.assert_frame_equal(result_csv, ref_csv)

    # ------------------------------------------------------

    def test_process_mergeContextNoCpG(self, tmpdir):
        output_prefix = f"{tmpdir}/output_mergeContextNoCpG"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_mergeContextNoCpG.run(
            [
                "process_mergeContextNoCpG",
                "--input_chg",
                f"{self.inputs_dir}/input_mergeContextNoCpG_CHG.bedGraph",
                "--input_chh",
                f"{self.inputs_dir}/input_mergeContextNoCpG_CHH.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        ref_csv = pd.read_csv(open(f"{self.inputs_dir}/ProcessMethylDackelMergeContextNoCpG.csv"))

        pd.testing.assert_frame_equal(result_csv, ref_csv)

    # ------------------------------------------------------

    def test_process_mergeContext(self, tmpdir):
        output_prefix = f"{tmpdir}/output_mergeContext"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        process_mergeContext.run(
            [
                "process_mergeContext",
                "--input",
                f"{self.inputs_dir}/input_mergeContext.bedGraph",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_file)
        pat = r"^hg"
        idx = result_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            result_csv_sub = result_csv.loc[idx, :].copy()

        ref_csv = pd.read_csv(open(f"{self.inputs_dir}/ProcessConcatMethylDackelMergeContext.csv"))
        idx = ref_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            ref_csv_sub = ref_csv.loc[idx, :].copy()

        pat = r"^PercentMethylation_"
        idx = result_csv_sub.metric.str.contains(pat)
        if idx.any(axis=None):
            result_output = result_csv_sub.loc[idx, :].copy()

        idx = ref_csv_sub.metric.str.contains(pat)
        if idx.any(axis=None):
            ref_output = ref_csv_sub.loc[idx, :].copy()

        assert np.all(np.sum(result_output.value) == np.sum(ref_output.value))

    # ------------------------------------------------------

    def test_concat_methyldackel_csvs(self, tmpdir):
        output_prefix = f"{tmpdir}/concat_methyldackel_csvs"
        output_csv_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

        input_file_name = f"{self.inputs_dir}/csv_files.txt"
        csv_files = open(input_file_name).read().rstrip().split(",")
        csv_files_amended = []
        for c in csv_files:
            csv_files_amended.append(f"{self.inputs_dir}/" + os.path.basename(c))
        csv_output = ",".join(csv_files_amended)
        out_file_name = f"{self.inputs_dir}/" + "csv_files_amended.txt"

        fobj = open(out_file_name, "w", encoding="utf-8")
        fobj.write(csv_output)
        fobj.close()

        concat_methyldackel_csvs.run(
            [
                "concat_methyldackel_csvs",
                "--input",
                f"{self.inputs_dir}/csv_files_amended.txt",
                "--output",
                f"{output_prefix}",
            ]
        )

        result_csv = pd.read_csv(output_csv_file)

        input_file_name = "ProcessConcatMethylDackelMergeContext.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))
        total = input_csv.shape[0]

        input_file_name = "ProcessMethylDackelMergeContextNoCpG.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))
        total += input_csv.shape[0]

        input_file_name = "ProcessMethylDackelPerRead.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))
        total += input_csv.shape[0]

        input_file_name = "ProcessMethylDackelMbias.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))
        total += input_csv.shape[0]
        os.unlink(out_file_name)
        assert result_csv.shape[0] == total

    # ------------------------------------------------------

    def test_methyldackel_utils_calc_percent_methylation(self, tmpdir):
        output_prefix = f"{tmpdir}/methyldackel_utils_pcnt_meth"
        output_file = output_prefix + ".csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        in_file_name = f"{self.inputs_dir}/input_mergeContext.bedGraph"
        col_names = ["chr", "start", "end", "PercentMethylation", "coverage_methylated", "coverage_unmethylated"]
        df_in_report = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)
        df_in_report["Coverage"] = df_in_report["coverage_methylated"] + df_in_report["coverage_unmethylated"]

        rel = False
        key_word = "hg"
        data_frame = pd.DataFrame()
        pat = r"^chr[0-9]+\b"
        idx = df_in_report.chr.str.contains(pat)
        if idx.any(axis=None):
            data_frame = df_in_report.loc[idx, :].copy()

        result_calc = calc_percent_methylation(key_word, data_frame, rel)

        input_file_name = "ProcessConcatMethylDackelMergeContext.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))

        pat = r"^hg"
        idx = input_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            ref_csv = input_csv.loc[idx, :].copy()

        pat = r"^PercentMethylation|TotalCpGs"
        idx = ref_csv.metric.str.contains(pat)
        if idx.any(axis=None):
            ref_csv_output = ref_csv.loc[idx, :].copy()

        assert np.all(np.sum(ref_csv_output.value) == np.sum(result_calc.value))

    # ------------------------------------------------------

    def test_methyldackel_utils_calc_coverage_methylation(self, tmpdir):

        in_file_name = f"{self.inputs_dir}/input_mergeContextNoCpG_CHG.bedGraph"
        col_names = ["chr", "start", "end", "PercentMethylation", "coverage_methylated", "coverage_unmethylated"]
        df_chg_input = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)

        # calculate total coverage
        df_chg_input["Coverage"] = df_chg_input.apply(
            lambda x: x["coverage_methylated"] + x["coverage_unmethylated"], axis=1
        )
        # remove non chr1-22 chromosomes
        pat = r"^chr[0-9]+\b"  # remove non
        idx = df_chg_input.chr.str.contains(pat)

        if idx.any(axis=None):
            df_chg = df_chg_input.loc[idx, :].copy()

        result_calc = calc_coverage_methylation("CHG", df_chg, True)

        input_file_name = "ProcessMethylDackelMergeContextNoCpG.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        input_csv = pd.read_csv(open(input_file_name))
        ref_csv = pd.DataFrame()

        pat = r"CHG"
        idx = input_csv.detail.str.contains(pat)
        if idx.any(axis=None):
            ref_csv = input_csv.loc[idx, :].copy()

        pat = r"Coverage"
        idx = ref_csv.metric.str.contains(pat)
        if idx.any(axis=None):
            ref_csv = ref_csv.loc[idx, :]

        assert np.all(np.sum(result_calc.value) == np.sum(ref_csv.value))

    # ------------------------------------------------------

    def test_methyldackel_utils_total_cpgs(self, tmpdir):

        in_file_name = f"{self.inputs_dir}/input_perRead.bedGraph"

        col_names = ["read_name", "chr", "start", "PercentMethylation", "TotalCpGs"]
        df_perRead = pd.read_csv(in_file_name, sep="\t", header=0, names=col_names)
        df_perRead.drop(columns="read_name", inplace=True)
        df_perRead.dropna(inplace=True)

        df_pcnt_meth = calc_percent_methylation("PercentMethylation", df_perRead, True)
        df_total_cpgs = calc_TotalCpGs("TotalCpGs", df_perRead)
        result_calc = pd.concat([df_pcnt_meth, df_total_cpgs], axis=0, ignore_index=True)

        input_file_name = "ProcessMethylDackelPerRead.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        ref_csv = pd.read_csv(open(input_file_name))

        assert np.allclose(np.sum(result_calc.value), np.sum(ref_csv.value))

    # ------------------------------------------------------

    def test_methyldackel_utils_get_dict(self, tmpdir):

        input_file_name = "ProcessConcatMethylDackelMergeContext.csv"
        input_file_name = f"{self.inputs_dir}/" + input_file_name
        ref_csv = pd.read_csv(open(input_file_name))

        dict_json_output = {}
        for detail in ref_csv["detail"].unique():
            temp_dict = get_dict_from_dataframe(ref_csv, detail)
            dict_json_output.update(temp_dict)

        calc_json = {"metrics": {}}
        calc_json["metrics"] = {"MergeContext": dict_json_output}

        input_file_name = "ProcessConcatMethylDackelMergeContext.json"
        input_file_name = f"{self.inputs_dir}/" + input_file_name

        ref_json = json.load(open(input_file_name))

        assert len(calc_json["metrics"]["MergeContext"]["hg"]) == len(ref_json["metrics"]["MergeContext"]["hg"])


# ------------------------------------------------------

# ------------------------------------------------------
