from test import get_resource_dir, test_dir

import pysam
import pytest
import json

from ugvc.pipelines.comparison import quick_fingerprinting

__inputs_dir = get_resource_dir(__file__)
__general_inputs_dir = f"{test_dir}/resources/general/chr1_head"


def test_quick_fingerprinting(tmpdir):
    fp_conf = {}
    fp_conf["cram_files"] = {
        "HG001": [f"{__inputs_dir}/034257-NA12878-Z0113-CAGTTCATCTGTGAT.chr1_head.cram"],
    }
    fp_conf["ground_truth_vcf_files"] = {
        "HG001": f"{__inputs_dir}/HG001_gt.vcf.gz",
    }
    fp_conf["ground_truth_hcr_files"] = {
        "HG001": f"{__inputs_dir}/HG001_hcr.bed",
    }
    fp_conf["references"] = {
        "ref_fasta": f"{__general_inputs_dir}/Homo_sapiens_assembly38.fasta",
        "ref_dict": f"{__general_inputs_dir}/Homo_sapiens_assembly38.dict",
        "ref_fasta_index": f"{__general_inputs_dir}/Homo_sapiens_assembly38.fasta.fai"
    }
    __conf = f"{tmpdir}/fingerprinting_conf.json"
    with open(__conf, 'w') as out:
        json.dump(fp_conf, out, indent=4)

    quick_fingerprinting.run(
        [
            "quick_fingerprinting",
            "--json_conf", __conf,
            "--region_str", "chr1:700000-800000",
            "--add_aws_auth_command", 
            "--out_dir", str(tmpdir)
        ]
    )

    output = f'{tmpdir}/quick_fingerprinting_results.txt'
    import sys
    with open(output) as out:
        lines = out.readlines()
        last_line = lines[-1]
        arr = last_line.split(' ')
        key, val = arr[-1].split('=')
        assert key == 'hit_fraction'
        assert float(val) > 0.99, 'hit fraction of HG001 vs itself is less than 0.99'