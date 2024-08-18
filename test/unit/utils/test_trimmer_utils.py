import filecmp
from os.path import basename, exists
from os.path import join as pjoin
from test.unit.utils.test_metrics_utils import inputs_dir

from ugbio_core import trimmer_utils


def test_merge_trimmer_histograms(tmpdir):
    suf = "A_hmer_start.T_hmer_start.A_hmer_end.T_hmer_end.native_adapter_with_leading_C.histogram.csv"
    trimmer_histograms = [
        pjoin(
            inputs_dir,
            f"029917001_1_Z0098-{suf}",
        ),
        pjoin(
            inputs_dir,
            f"029917001_2_Z0098-{suf}",
        ),
    ]
    expected_output = pjoin(
        inputs_dir,
        f"EXPECTED.029917001_Z0098-{suf}",
    )
    output_path = pjoin(
        tmpdir,
        basename(trimmer_histograms[0]),
    )
    merged_histogram = trimmer_utils.merge_trimmer_histograms(trimmer_histograms=trimmer_histograms, output_path=tmpdir)
    assert exists(merged_histogram)
    assert merged_histogram == output_path
    assert filecmp.cmp(merged_histogram, expected_output)

    output_path = pjoin(
        tmpdir,
        basename(trimmer_histograms[1]),
    )
    merged_histogram = trimmer_utils.merge_trimmer_histograms(
        trimmer_histograms=trimmer_histograms[1], output_path=tmpdir
    )
    assert exists(merged_histogram)
    assert filecmp.cmp(merged_histogram, trimmer_histograms[1])