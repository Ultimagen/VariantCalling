import os
import subprocess
from collections import Counter
from os.path import exists
from os.path import join as pjoin

from simppl.simple_pipeline import SimplePipeline

from test import get_resource_dir, test_dir

import pandas as pd
import pysam

import ugvc.comparison.vcf_pipeline_utils as vcf_pipeline_utils

inputs_dir = get_resource_dir(__file__)
common_dir = pjoin(test_dir, "resources", "general")


def test_fix_errors():
    data = pd.read_hdf(pjoin(inputs_dir, "h5_file_unitest.h5"), key="concordance")
    df = vcf_pipeline_utils._fix_errors(data)
    assert all(
        df[((df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna())))][
            "gt_ground_truth"
        ].eq(
            df[(df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna()))][
                "gt_ultima"
            ]
        )
    )

    # (None, TP) (None,FN_CA)
    assert (
        df[(df["call"].isna()) & ((df["base"] == "TP") | (df["base"] == "FN_CA"))].size
        == 0
    )
    # (FP_CA,FN_CA), (FP_CA,None)
    temp_df = df.loc[
        (df["call"] == "FP_CA") & ((df["base"] == "FN_CA") | (df["base"].isna())),
        ["gt_ultima", "gt_ground_truth"],
    ]
    assert all(
        temp_df.apply(
            lambda x: (
                (x["gt_ultima"][0] == x["gt_ground_truth"][0])
                & (x["gt_ultima"][1] != x["gt_ground_truth"][1])
            )
            | (
                (x["gt_ultima"][1] == x["gt_ground_truth"][1])
                & (x["gt_ultima"][0] != x["gt_ground_truth"][0])
            )
            | (
                (x["gt_ultima"][0] == x["gt_ground_truth"][1])
                & (x["gt_ultima"][1] != x["gt_ground_truth"][0])
            )
            | (
                (x["gt_ultima"][1] == x["gt_ground_truth"][0])
                & (x["gt_ultima"][0] != x["gt_ground_truth"][1])
            ),
            axis=1,
        )
    )


class TestVCF2Concordance:
    def test_qual_not_nan(self):
        input_vcf = pjoin(inputs_dir, "chr2.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "chr2.conc.vcf.gz")
        result = vcf_pipeline_utils.vcf2concordance(
            input_vcf, concordance_vcf, "VCFEVAL"
        )
        assert pd.isnull(result.query("classify!='fn'").qual).sum() == 0
        assert pd.isnull(result.query("classify!='fn'").sor).sum() == 0

    def test_filtered_out_missing(self):
        input_vcf = pjoin(inputs_dir, "hg002.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.conc.vcf.gz")
        result = vcf_pipeline_utils.vcf2concordance(
            input_vcf, concordance_vcf, "VCFEVAL"
        )
        assert ((result["call"] == "IGN") & (pd.isnull(result["base"]))).sum() == 0

    def test_filtered_out_tp_became_fn(self):
        input_vcf = pjoin(inputs_dir, "hg002.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.conc.vcf.gz")
        result = vcf_pipeline_utils.vcf2concordance(
            input_vcf, concordance_vcf, "VCFEVAL"
        )
        assert ((result["call"] == "IGN") & (result["base"] == "FN")).sum() > 0
        take = result[(result["call"] == "IGN") & (result["base"] == "FN")]
        assert (take["classify"] == "fn").all()

    def test_excluded_regions_are_ignored(self):
        input_vcf = pjoin(inputs_dir, "hg002.excluded.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.excluded.conc.vcf.gz")
        result = vcf_pipeline_utils.vcf2concordance(
            input_vcf, concordance_vcf, "VCFEVAL"
        )
        assert ((result["call"] == "OUT")).sum() == 0
        assert ((result["base"] == "OUT")).sum() == 0

    def test_all_ref_never_false_negative(self):
        input_vcf = pjoin(inputs_dir, "hg002.allref.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.allref.conc.vcf.gz")
        result = vcf_pipeline_utils.vcf2concordance(
            input_vcf, concordance_vcf, "VCFEVAL"
        )
        calls = result[result["gt_ground_truth"] == (0, 0)].classify_gt.value_counts()
        assert "fn" not in calls.index


class TestVCFevalRun:
    ref_genome = pjoin(common_dir, "sample.fasta")
    sample_calls = pjoin(inputs_dir, "sample.sd.vcf.gz")
    truth_calls = pjoin(inputs_dir, "gtr.sample.sd.vcf.gz")
    high_conf = pjoin(inputs_dir, "highconf.interval_list")

    def test_vcfeval_run_ignore_filter(self, tmp_path):
        sp = SimplePipeline(0, 100, False)
        vcf_pipeline_utils.run_vcfeval_concordance(
            sp=sp,
            input_file=self.sample_calls,
            truth_file=self.truth_calls,
            output_prefix=str(tmp_path / "sample.ignore_filter"),
            ref_genome=self.ref_genome,
            comparison_intervals=self.high_conf,
            input_sample="sm1",
            truth_sample="HG001",
            ignore_filter=True,
        )
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(
            str(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")
        ) as vcf:
            calls = Counter([x.info["CALL"] for x in vcf])
        assert calls == {"FP": 99, "TP": 1}

    def test_vcfeval_run_use_filter(self, tmp_path):
        sp = SimplePipeline(0, 100, False)
        vcf_pipeline_utils.run_vcfeval_concordance(
            sp=sp,
            input_file=self.sample_calls,
            truth_file=self.truth_calls,
            output_prefix=str(tmp_path / "sample.use_filter"),
            ref_genome=self.ref_genome,
            comparison_intervals=self.high_conf,
            input_sample="sm1",
            truth_sample="HG001",
            ignore_filter=False,
        )
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(
            str(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")
        ) as vcf:
            calls = Counter([x.info["CALL"] for x in vcf])
        assert calls == {"FP": 91, "TP": 1, "IGN": 8}


def test_intersect_bed_files(mocker, tmp_path):
    bed1 = pjoin(inputs_dir, "bed1.bed")
    bed2 = pjoin(inputs_dir, "bed2.bed")
    output_path = pjoin(tmp_path, "output.bed")

    #spy_subprocess = mocker.spy(subprocess, "call")
    sp = SimplePipeline(0, 10, False)
    vcf_pipeline_utils.intersect_bed_files(sp, bed1, bed2, output_path)
    #spy_subprocess.assert_called_once_with(
    #    ["bedtools", "intersect", "-a", bed1, "-b", bed2], stdout=mocker.ANY
    #)

    assert exists(output_path)


def test_bed_file_length():
    bed1 = pjoin(inputs_dir, "bed1.bed")
    result = vcf_pipeline_utils.bed_file_length(bed1)
    assert result == 3026


def test_IntervalFile_init_bed_input():
    bed1 = pjoin(inputs_dir, "bed1.bed")
    ref_genome = pjoin(common_dir, "sample.fasta")
    interval_list_path = pjoin(inputs_dir, "bed1.interval_list")

    sp = SimplePipeline(0, 100, False)
    intervalFile = vcf_pipeline_utils.IntervalFile(sp, bed1, ref_genome, None)

    assert intervalFile.as_bed_file() == bed1
    assert intervalFile.as_interval_list_file() == interval_list_path
    assert exists(interval_list_path)
    assert not intervalFile.is_none()
    os.remove(interval_list_path)


def test_IntervalFile_init_interval_list_input(mocker):
    interval_list = pjoin(inputs_dir, "interval_list1.interval_list")
    ref_genome = pjoin(common_dir, "sample.fasta")
    bed_path = pjoin(inputs_dir, "interval_list1.bed")

    sp = SimplePipeline(0, 100, False)
    intervalFile = vcf_pipeline_utils.IntervalFile(sp, interval_list, ref_genome, None)

    assert intervalFile.as_bed_file() == bed_path
    assert intervalFile.as_interval_list_file() == interval_list
    assert exists(bed_path)
    assert not intervalFile.is_none()
    os.remove(bed_path)


def test_IntervalFile_init_error():
    ref_genome = pjoin(common_dir, "sample.fasta")
    sp = SimplePipeline(0, 100, False)
    intervalFile = vcf_pipeline_utils.IntervalFile(sp, ref_genome, ref_genome, None)
    assert intervalFile.as_bed_file() is None
    assert intervalFile.as_interval_list_file() is None
    assert intervalFile.is_none()
