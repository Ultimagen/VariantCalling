import os
from collections import defaultdict
from os.path import join as pjoin
from test import get_resource_dir

import pysam
import pytest

from ugvc.mrd.featuremap_consensus_utils import pileup_featuremap, pileup_featuremap_on_an_interval_list

inputs_dir = get_resource_dir(__file__)


@pytest.mark.parametrize(
    "genomic_interval, interval_list, expected_num_variants",
    [
        ("chr19:1-400000", None, 66),
        (None, None, 477),
        ("chr2:1-1000", None, 0),
        (None, pjoin(inputs_dir, "scattered.interval_list"), 477),
    ],
)
def test_pileup_featuremap(
    tmpdir,
    genomic_interval,
    interval_list,
    expected_num_variants,
):
    # create input featuremap vcf file
    input_featuremap_vcf = pjoin(inputs_dir, "HG001_HG002_tumor_tumor_in_normal_31977.gtr.hcr.chr19_1-1000000.vcf.gz")

    # call the function with different arguments
    pileup_featuremap_vcf = pjoin(tmpdir, "pileup_featuremap.vcf.gz")
    if interval_list is None:
        pileup_featuremap(
            featuremap=input_featuremap_vcf,
            output_vcf=pileup_featuremap_vcf,
            genomic_interval=genomic_interval,
        )
    else:
        pileup_featuremap_on_an_interval_list(
            featuremap=input_featuremap_vcf,
            output_vcf=pileup_featuremap_vcf,
            interval_list=interval_list,
        )

    # check that the output file exists and has the expected content
    assert os.path.isfile(pileup_featuremap_vcf)
    # count the number of variants (excluding the header)
    cons_dict = defaultdict(dict)
    for rec in pysam.VariantFile(pileup_featuremap_vcf):
        rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        if rec_id not in cons_dict:
            cons_dict[rec_id]["count"] = 0
        cons_dict[rec_id]["count"] += 1
    # check the number of variants
    num_variants = len(cons_dict)
    assert num_variants == expected_num_variants
    # assert a single entry per variant
    assert all(cons["count"] == 1 for cons in cons_dict.values())
