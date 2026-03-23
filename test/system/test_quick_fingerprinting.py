from test import get_resource_dir, test_dir

import os
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
    with open(output) as out:
        lines = out.readlines()
        last_line = lines[-1]
        arr = last_line.split(' ')
        key, val = arr[-1].split('=')
        assert key == 'hit_fraction'
        assert float(val) > 0.99, 'hit fraction of HG001 vs itself is less than 0.99'


def _make_conf(inputs_dir, general_inputs_dir):
    """Return a fingerprinting config dict pointing at the test resources."""
    return {
        "cram_files": {
            "HG001": [f"{inputs_dir}/034257-NA12878-Z0113-CAGTTCATCTGTGAT.chr1_head.cram"],
        },
        "ground_truth_vcf_files": {
            "HG001": f"{inputs_dir}/HG001_gt.vcf.gz",
        },
        "ground_truth_hcr_files": {
            "HG001": f"{inputs_dir}/HG001_hcr.bed",
        },
        "references": {
            "ref_fasta": f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta",
            "ref_dict": f"{general_inputs_dir}/Homo_sapiens_assembly38.dict",
            "ref_fasta_index": f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta.fai",
        },
    }


def _read_last_hit_fraction(out_dir):
    output = f"{out_dir}/quick_fingerprinting_results.txt"
    with open(output) as fh:
        lines = fh.readlines()
    last_line = lines[-1]
    key, val = last_line.split()[-1].split("=")
    assert key == "hit_fraction"
    return float(val)


def test_quick_fingerprinting_with_regions_bed(tmpdir):
    """Fingerprinting with --regions_bed: should still match HG001 to itself."""
    fp_conf = _make_conf(__inputs_dir, __general_inputs_dir)
    conf_path = f"{tmpdir}/fingerprinting_conf.json"
    with open(conf_path, "w") as fh:
        json.dump(fp_conf, fh, indent=4)

    quick_fingerprinting.run(
        [
            "quick_fingerprinting",
            "--json_conf", conf_path,
            "--region_str", "chr1:700000-812260",
            "--regions_bed", f"{__inputs_dir}/exome_regions.bed",
            "--out_dir", str(tmpdir),
        ]
    )

    hit_fraction = _read_last_hit_fraction(str(tmpdir))
    assert hit_fraction > 0.99, f"hit fraction with regions_bed is {hit_fraction}, expected > 0.99"
    # Confirm the intersected bed was created
    assert os.path.exists(f"{tmpdir}/regions_bed_in_region.bed"), "regions_bed_in_region.bed was not created"


def test_quick_fingerprinting_with_regions_bed_chrom_only(tmpdir):
    """Fingerprinting with --regions_bed and a bare chromosome region_str (grep path)."""
    fp_conf = _make_conf(__inputs_dir, __general_inputs_dir)
    conf_path = f"{tmpdir}/fingerprinting_conf.json"
    with open(conf_path, "w") as fh:
        json.dump(fp_conf, fh, indent=4)

    quick_fingerprinting.run(
        [
            "quick_fingerprinting",
            "--json_conf", conf_path,
            "--region_str", "chr1",
            "--regions_bed", f"{__inputs_dir}/exome_regions.bed",
            "--out_dir", str(tmpdir),
        ]
    )

    hit_fraction = _read_last_hit_fraction(str(tmpdir))
    assert hit_fraction > 0.99, f"hit fraction (chrom-only + regions_bed) is {hit_fraction}, expected > 0.99"
    assert os.path.exists(f"{tmpdir}/regions_bed_in_region.bed"), "regions_bed_in_region.bed was not created"