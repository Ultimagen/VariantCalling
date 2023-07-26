# %%
import os
import sys

for path in [pjoin(os.environ["HOME"], "proj/VariantCalling"), "/VariantCalling"]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(1, path)
test_dir = "/home/itai.rusinek/proj/VariantCalling/test/resources/general"
inputs_dir = "/home/itai.rusinek/proj/VariantCalling/test/resources/unit/mrd/test_mrd_utils"

from os.path import join as pjoin

import pandas as pd
from pandas.testing import assert_frame_equal

from ugvc.mrd.balanced_epcr_utils import read_trimmer_histogram

# from test import get_resource_dir, test_dir


general_inputs_dir = pjoin(test_dir, "resources", "general")
# inputs_dir = get_resource_dir(__file__)
input_histogram_csv = pjoin(
    inputs_dir,
    "130713_UGAv3-51.trimming.A_hmer_5.T_hmer_5.A_hmer_3.T_hmer_3.native_adapter_with_leading_C.histogram.csv",
)


def test_read_trimmer_histogram():
    df_trimmer_histogram = read_trimmer_histogram(input_histogram_csv)
    return trimmer_histogram
    # assert_frame_equal(
    #     parsed_intersection_dataframe2.reset_index(),
    #     parsed_intersection_dataframe_expected,
    # )


# %%
df_trimmer_histogram = test_read_trimmer_histogram()
df_trimmer_histogram.to_csv()
