import os
import subprocess
import tempfile
from typing import Optional

import pandas as pd
from pandas import DataFrame

authorize_gcp_command = 'export GCS_OAUTH_TOKEN=`gcloud auth application-default print-access-token`'


def is_gcs_url(path: str) -> bool:
    return path.startswith('gs://')


def read_hdf(path: str, key: Optional[str]) -> DataFrame:
    if is_gcs_url(path):
        local_h5_file = f'{tempfile.gettempprefix()}.h5'
        subprocess.run(f'{authorize_gcp_command}; gsutil cp {path} {local_h5_file}')
        df = pd.read_hdf(local_h5_file, key=key)
        os.remove(local_h5_file)
    else:
        df = pd.read_hdf(path)
    return df

