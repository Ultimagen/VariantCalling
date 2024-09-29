import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir

import mock
import pandas as pd
from ugbio_core.consts import FileExtension

from ugvc.pipelines import run_no_gt_report

inputs_dir = get_resource_dir(__file__)
general_inputs_dir = f"{test_dir}/resources/general/"


def test_allele_freq_hist():
    df = pd.DataFrame(
        {
            "af": [0.1, 0.2, 0.3, 0.4, 0.5],
            "variant_type": ["snp", "snp", "h-indel", "non-h-indel", "non-h-indel"],
        }
    )
    result = run_no_gt_report.allele_freq_hist(df)
    assert len(result.keys()) == 3
    assert len(result["snp"]) == 100
    assert len(result["h-indel"]) == 100
    assert len(result["non-h-indel"]) == 100
    assert result["snp"].iloc[10] == 1
    assert result["snp"].iloc[20] == 1
    assert sum(result["snp"] == 0) == 98


@mock.patch("subprocess.check_call")
def test_variant_eval_statistics(mocked_subprocess):
    output_prefix = pjoin(inputs_dir, "collect_stats_unfiltered")
    data = run_no_gt_report.variant_eval_statistics(
        "vcf", "ref", "db_snp", output_prefix, 0, ["exome", "extended_exome", "high_conf"], append_confident_calls=True
    )
    subprocess.check_call.assert_called_once_with(
        [
            "gatk",
            "VariantEval",
            "--eval",
            "vcf",
            "--reference",
            "ref",
            "--dbsnp",
            "db_snp",
            "--output",
            f"{output_prefix}{FileExtension.TXT.value}",
            "--selectNames",
            "exome",
            "--selectNames",
            "extended_exome",
            "--selectNames",
            "high_conf",
            "--selectNames",
            "CONFIDENT_CALLS_GQ20",
            "--select",
            'vc.hasAttribute("exome")',
            "--select",
            'vc.hasAttribute("extended_exome")',
            "--select",
            'vc.hasAttribute("high_conf")',
            "--select",
            "vc.getGenotype(0).getGQ()>=20",
        ]
    )
    for name in [
        "CompOverlap",
        "CountVariants",
        "IndelLengthHistogram",
        "IndelSummary",
        "MetricsCollection",
        "TiTvVariantEvaluator",
        "ValidationReport",
        "VariantSummary",
    ]:
        assert data[name] is not None
        assert isinstance(data[name], pd.DataFrame)


def test_insertion_deletion_statistics():
    df_path = pjoin(inputs_dir, f"df{FileExtension.HDF.value}")
    df = pd.read_hdf(df_path, "concordance")
    result = run_no_gt_report.insertion_deletion_statistics(df)
    assert result["homo"] is not None
    assert result["hete"] is not None

    assert result["homo"].shape == (4, 12)
    assert result["hete"].shape == (4, 12)

    assert result["homo"].loc["ins G", 2] == 1
    result["homo"]["ins G", 2] = 0
    assert all(result["hete"] == 0)
    assert all(result["hete"] == 0)


def test_snp_statistics():
    ref_fasta = pjoin(general_inputs_dir, f"sample{FileExtension.FASTA.value}")
    df_path = pjoin(inputs_dir, f"df{FileExtension.HDF.value}")
    df = pd.read_hdf(df_path, "chr20")
    result = run_no_gt_report.snp_statistics(df, ref_fasta)

    assert len(result) == 96
    assert result.loc[("CCA", "A")] == 1
    assert result.loc[("CCA", "T")] == 1
    assert result.loc[("AAT", "G")] == 1
    assert result.loc[("ACT", "T")] == 2
