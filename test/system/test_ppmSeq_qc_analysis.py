from os.path import join as pjoin
from test import get_resource_dir

from ugvc.pipelines.mrd import ppmSeq_qc_analysis


def test_ppmSeq_analysis_ppmSeq_legacy_v5(tmpdir):
    resource_dir = get_resource_dir(__file__)
    trimmer_histogram = pjoin(
        resource_dir,
        "A_hmer_start.T_hmer_start.A_hmer_end.T_hmer_end.native_adapter_with_leading_C.histogram.csv",
    )
    trimmer_failure_codes = pjoin(
        resource_dir,
        "ppmSeq.healthy.022782-Lb_2146-UGAv3-168.failure_codes.csv",
    )
    sorter_csv = pjoin(resource_dir, "ppmSeq.healthy.022782-Lb_2146-UGAv3-168.csv")
    sorter_json = pjoin(resource_dir, "ppmSeq.healthy.022782-Lb_2146-UGAv3-168.json")

    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "legacy_v5",
            "--trimmer-histogram-csv",
            trimmer_histogram,
            "--trimmer-failure-codes-csv",
            trimmer_failure_codes,
            "--sorter-stats-csv",
            sorter_csv,
            "--sorter-stats-json",
            sorter_json,
            "--output-path",
            tmpdir.dirname,
            "--output-basename",
            "ppmSeq.healthy.022782-Lb_2146-UGAv3-168",
            "--legacy-histogram-column-names",
        ]
    )


def test_ppmSeq_analysis_ppmSeq_v1(tmpdir):
    resource_dir = get_resource_dir(__file__)
    trimmer_histogram = pjoin(
        resource_dir,
        "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT."
        "Start_loop.Start_loop.End_loop.End_loop.native_adapter.histogram.csv",
    )
    trimmer_failure_codes = pjoin(
        resource_dir,
        "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.failure_codes.csv",
    )
    sorter_csv = pjoin(resource_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.csv")
    sorter_json = pjoin(resource_dir, "037239-CgD1502_Cord_Blood-Z0032-CTCTGTATTGCAGAT.json")

    ppmSeq_qc_analysis.run(
        [
            "ppmSeq_qc_analysis",
            "--adapter-version",
            "v1",
            "--trimmer-histogram-csv",
            trimmer_histogram,
            "--trimmer-failure-codes-csv",
            trimmer_failure_codes,
            "--sorter-stats-csv",
            sorter_csv,
            "--sorter-stats-json",
            sorter_json,
            "--output-path",
            tmpdir.dirname,
            "--output-basename",
            "037239-CgD1502_Cord_Blood-Z0032",
            "--legacy-histogram-column-names",
        ]
    )
