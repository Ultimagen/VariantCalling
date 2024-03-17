from os.path import join as pjoin
from test import get_resource_dir

from ugvc.pipelines.mrd import balanced_strand_analysis


def test_balanced_strand_analysis_ppmSeq_v1(tmpdir):
    resource_dir = get_resource_dir(__file__)
    trimmer_histogram = pjoin(
        resource_dir,
        "A_hmer_start.T_hmer_start.A_hmer_end.T_hmer_end.native_adapter_with_leading_C.histogram.csv",
    )
    trimmer_failure_codes = pjoin(
        resource_dir,
        "balanced_strand.healthy.022782-Lb_2146-UGAv3-168.failure_codes.csv",
    )
    sorter_csv = pjoin(resource_dir, "balanced_strand.healthy.022782-Lb_2146-UGAv3-168.csv")
    sorter_json = pjoin(resource_dir, "balanced_strand.healthy.022782-Lb_2146-UGAv3-168.json")

    balanced_strand_analysis.run(
        [
            "balanced_strand_analysis",
            "--adapter-version",
            "LA_v5and6",
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
            "balanced_strand.healthy.022782-Lb_2146-UGAv3-168",
            "--legacy-histogram-column-names",
        ]
    )


def test_balanced_strand_analysis_ppmSeq_v2(tmpdir):
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

    balanced_strand_analysis.run(
        [
            "balanced_strand_analysis",
            "--adapter-version",
            "LA_v7",
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
