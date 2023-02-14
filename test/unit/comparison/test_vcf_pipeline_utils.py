import os
import shutil
import subprocess
from collections import Counter
from os.path import basename, exists
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import pandas as pd
import pysam
from simppl.simple_pipeline import SimplePipeline

from ugvc.comparison.concordance_utils import read_hdf
from ugvc.comparison.vcf_pipeline_utils import VcfPipelineUtils, _fix_errors, bed_file_length, vcf2concordance
from ugvc.vcfbed import vcftools
from ugvc.vcfbed.interval_file import IntervalFile

inputs_dir = get_resource_dir(__file__)
common_dir = pjoin(test_dir, "resources", "general")


def test_fix_errors():
    data = read_hdf(pjoin(inputs_dir, "h5_file_unitest.h5"), key="concordance")
    df = _fix_errors(data)
    assert all(
        df[((df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna())))]["gt_ground_truth"].eq(
            df[(df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna()))]["gt_ultima"]
        )
    )

    # (None, TP) (None,FN_CA)
    pd.set_option("display.max_columns", None)
    assert df[(df["call"].isna()) & ((df["base"] == "TP") | (df["base"] == "FN_CA"))].size == 20
    # (FP_CA,FN_CA), (FP_CA,None)
    temp_df = df.loc[
        (df["call"] == "FP_CA") & ((df["base"] == "FN_CA") | (df["base"].isna())),
        ["gt_ultima", "gt_ground_truth"],
    ]
    assert all(
        temp_df.apply(
            lambda x: ((x["gt_ultima"][0] == x["gt_ground_truth"][0]) & (x["gt_ultima"][1] != x["gt_ground_truth"][1]))
            | ((x["gt_ultima"][1] == x["gt_ground_truth"][1]) & (x["gt_ultima"][0] != x["gt_ground_truth"][0]))
            | ((x["gt_ultima"][0] == x["gt_ground_truth"][1]) & (x["gt_ultima"][1] != x["gt_ground_truth"][0]))
            | ((x["gt_ultima"][1] == x["gt_ground_truth"][0]) & (x["gt_ultima"][0] != x["gt_ground_truth"][1])),
            axis=1,
        )
    )


def test_transform_hom_calls_to_het_calls(tmpdir):

    input_vcf = pjoin(inputs_dir, "dv.input.vcf.gz")
    vpu = VcfPipelineUtils()
    shutil.copyfile(input_vcf, pjoin(tmpdir, basename(input_vcf)))
    expected_output_file = pjoin(tmpdir, basename(input_vcf).replace(".vcf.gz", ".rev.hom.ref.vcf.gz"))
    expected_output_index_file = pjoin(tmpdir, basename(input_vcf).replace(".vcf.gz", ".rev.hom.ref.vcf.gz.tbi"))

    vpu.transform_hom_calls_to_het_calls(pjoin(tmpdir, basename(input_vcf)), expected_output_file)
    assert exists(expected_output_file)
    assert exists(expected_output_index_file)
    input_df = vcftools.get_vcf_df(input_vcf)
    select = (input_df["filter"] != "PASS") & ((input_df["gt"] == (0, 0)) | (input_df["gt"] == (None, None)))
    assert select.sum() > 0
    input_df = vcftools.get_vcf_df(expected_output_file)
    select = (input_df["filter"] != "PASS") & ((input_df["gt"] == (0, 0)) | (input_df["gt"] == (None, None)))
    assert select.sum() == 0


class TestVCF2Concordance:
    def test_qual_not_nan(self):
        input_vcf = pjoin(inputs_dir, "chr2.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "chr2.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        assert pd.isnull(result.query("classify!='fn'").qual).sum() == 0
        assert pd.isnull(result.query("classify!='fn'").sor).sum() == 0

    def test_filtered_out_missing(self):
        input_vcf = pjoin(inputs_dir, "hg002.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        assert ((result["call"] == "IGN") & (pd.isnull(result["base"]))).sum() == 0

    def test_filtered_out_tp_became_fn(self):
        input_vcf = pjoin(inputs_dir, "hg002.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        take = result[(result["call"] == "IGN") & (result["base"] == "FN")]
        assert take.shape[0] > 0
        assert (take["classify"] == "fn").all()

    def test_excluded_regions_are_ignored(self):
        input_vcf = pjoin(inputs_dir, "hg002.excluded.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.excluded.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        assert (result["call"] == "OUT").sum() == 0
        assert (result["base"] == "OUT").sum() == 0

    def test_all_ref_never_false_negative(self):
        input_vcf = pjoin(inputs_dir, "hg002.allref.vcf.gz")
        concordance_vcf = pjoin(inputs_dir, "hg002.allref.conc.vcf.gz")
        result = vcf2concordance(input_vcf, concordance_vcf)
        calls = result[result["gt_ground_truth"] == (0, 0)].classify_gt.value_counts()
        assert "fn" not in calls.index


class TestVCFEvalRun:
    ref_genome = pjoin(common_dir, "sample.fasta")
    sample_calls = pjoin(inputs_dir, "sample.sd.vcf.gz")
    truth_calls = pjoin(inputs_dir, "gtr.sample.sd.vcf.gz")

    def test_vcfeval_run_ignore_filter(self, tmp_path):
        sp = SimplePipeline(0, 100, False)
        high_conf = IntervalFile(None, pjoin(inputs_dir, "highconf.interval_list")).as_bed_file()
        VcfPipelineUtils(sp).run_vcfeval_concordance(
            input_file=self.sample_calls,
            truth_file=self.truth_calls,
            output_prefix=str(tmp_path / "sample.ignore_filter"),
            ref_genome=self.ref_genome,
            evaluation_regions=high_conf,
            comparison_intervals=high_conf,
            input_sample="sm1",
            truth_sample="HG001",
            ignore_filter=True,
        )
        os.remove(high_conf)
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(str(tmp_path / "sample.ignore_filter.vcfeval_concordance.vcf.gz")) as vcf:
            calls = Counter([x.info["CALL"] for x in vcf])
        assert calls == {"FP": 99, "TP": 1}

    def test_vcfeval_run_use_filter(self, tmp_path):
        sp = SimplePipeline(0, 100, False)
        high_conf = IntervalFile(None, pjoin(inputs_dir, "highconf.interval_list")).as_bed_file()
        VcfPipelineUtils(sp).run_vcfeval_concordance(
            input_file=self.sample_calls,
            truth_file=self.truth_calls,
            output_prefix=str(tmp_path / "sample.use_filter"),
            ref_genome=self.ref_genome,
            evaluation_regions=high_conf,
            comparison_intervals=high_conf,
            input_sample="sm1",
            truth_sample="HG001",
            ignore_filter=False,
        )
        os.remove(high_conf)
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")
        assert exists(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz.tbi")

        with pysam.VariantFile(str(tmp_path / "sample.use_filter.vcfeval_concordance.vcf.gz")) as vcf:
            calls = Counter([x.info["CALL"] for x in vcf])
        assert calls == {"FP": 91, "TP": 1, "IGN": 8}


def test_intersect_bed_files(mocker, tmp_path):
    bed1 = pjoin(inputs_dir, "bed1.bed")
    bed2 = pjoin(inputs_dir, "bed2.bed")
    output_path = pjoin(tmp_path, "output.bed")

    # Test with simple pipeline
    sp = SimplePipeline(0, 10, False)
    VcfPipelineUtils(sp).intersect_bed_files(bed1, bed2, output_path)

    # Test without simple pipeline
    spy_subprocess = mocker.spy(subprocess, "call")

    VcfPipelineUtils().intersect_bed_files(bed1, bed2, output_path)
    spy_subprocess.assert_called_once_with(["bedtools", "intersect", "-a", bed1, "-b", bed2], stdout=mocker.ANY)
    assert exists(output_path)


def test_bed_file_length():
    bed1 = pjoin(inputs_dir, "bed1.bed")
    result = bed_file_length(bed1)
    assert result == 3026
