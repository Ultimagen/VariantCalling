# %%
import os
import sys

for path in [
    os.path.join(os.environ["HOME"], "proj/VariantCalling"),
    "/VariantCalling",
]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(1, path)
test_dir = "/home/itai.rusinek/proj/VariantCalling/test/resources/general"
inputs_dir = "/home/itai.rusinek/proj/VariantCalling/test/resources/unit/mrd/test_mrd_utils"

import tempfile
from os.path import join as pjoin

import pandas as pd
from pandas.testing import assert_frame_equal

from ugvc.mrd.balanced_epcr_utils import read_balanced_epcr_trimmer_histogram

# from test import get_resource_dir, test_dir


general_inputs_dir = pjoin(test_dir, "resources", "general")
# inputs_dir = get_resource_dir(__file__)
input_histogram_LAv5and6_csv = pjoin(
    inputs_dir,
    "130713_UGAv3-51.trimming.A_hmer_5.T_hmer_5.A_hmer_3.T_hmer_3.native_adapter_with_leading_C.histogram.csv",
)
parsed_histogram_LAv5and6_parquet = pjoin(inputs_dir, "130713_UGAv3-51.parsed_histogram.parquet")
input_histogram_LAv5_csv = pjoin(inputs_dir, "130715_UGAv3-132.trimming.A_hmer.T_hmer.histogram.csv")
parsed_histogram_LAv5_parquet = pjoin(inputs_dir, "130715_UGAv3-132.parsed_histogram.parquet")


def test_read_balanced_epcr_LAv5and6_trimmer_histogram():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.parquet")
        df_trimmer_histogram = read_balanced_epcr_trimmer_histogram(
            input_histogram_LAv5and6_csv, output_filename=tmp_out_path
        )
        df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
        df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_LAv5and6_parquet)
        assert_frame_equal(
            df_trimmer_histogram,
            df_trimmer_histogram_expected,
        )
        assert_frame_equal(
            df_trimmer_histogram_from_parquet,
            df_trimmer_histogram_expected,
        )


def test_read_balanced_epcr_LAv5_trimmer_histogram():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_out_path = pjoin(tmpdirname, "tmp_out.parquet")
        df_trimmer_histogram = read_balanced_epcr_trimmer_histogram(
            input_histogram_LAv5_csv, output_filename=tmp_out_path
        )
        df_trimmer_histogram_from_parquet = pd.read_parquet(tmp_out_path)
        df_trimmer_histogram_expected = pd.read_parquet(parsed_histogram_LAv5_parquet)
        assert_frame_equal(
            df_trimmer_histogram,
            df_trimmer_histogram_expected,
        )
        assert_frame_equal(
            df_trimmer_histogram_from_parquet,
            df_trimmer_histogram_expected,
        )


# %%
test_read_balanced_epcr_LAv5and6_trimmer_histogram()
test_read_balanced_epcr_LAv5_trimmer_histogram()
# df_trimmer_histogram1 = test_read_balanced_epcr_LAv5and6_trimmer_histogram()
# df_trimmer_histogram1.to_parquet(parsed_histogram_LAv5and6_parquet)
# df_trimmer_histogram2 = test_read_balanced_epcr_LAv5_trimmer_histogram()
# df_trimmer_histogram2.to_parquet(parsed_histogram_LAv5_parquet)
