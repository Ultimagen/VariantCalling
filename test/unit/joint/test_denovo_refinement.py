from os.path import join as pjoin
from test import get_resource_dir

import pandas as pd

from ugvc.joint.denovo_refinement import add_paternal_qualities_to_denovo_vcf, get_parental_vcf_df


def test_get_parental_vcf_df():
    datadir = get_resource_dir(__file__)
    maternal_dict = {"sample1": pjoin(datadir, "sample1_maternal.vcf.gz")}
    paternal_dict = {"sample1": pjoin(datadir, "sample1_paternal.vcf.gz")}
    df = get_parental_vcf_df(maternal_dict, paternal_dict)
    expected = pd.read_hdf(pjoin(datadir, "parental_vcf_df.h5"), key="df")
    assert df.equals(expected)


def test_add_paternal_qualities_to_denovo_vcf():
    datadir = get_resource_dir(__file__)
    denovo_vcf = pjoin(datadir, "denovo.vcf.gz")
    parental_vcf_df = pd.DataFrame(pd.read_hdf(pjoin(datadir, "parental_vcf_df.h5"), key="df"))
    parental_vcf_df.rename(
        {"sample1-mother": "CL10370-mother", "sample1-father": "CL10370-father"}, axis=1, level=0, inplace=True
    )
    df = add_paternal_qualities_to_denovo_vcf(denovo_vcf, parental_vcf_df)
    expected = pd.read_hdf(pjoin(datadir, "denovo_vcf_with_qual.h5"), key="df")
    assert df.equals(expected)
