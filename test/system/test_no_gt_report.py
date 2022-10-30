import shutil
import subprocess
from os.path import join as pjoin
from test import get_resource_dir, test_dir


def test_no_gt_report(tmpdir):
    path = test_dir
    datadir = get_resource_dir(__file__)
    report_path = pjoin(path, "..", "ugvc", "reports")

    shutil.copy(pjoin(report_path, "report_wo_gt.ipynb"), tmpdir)
    shutil.copy(pjoin(report_path, "nexusplt.py"), tmpdir)
    shutil.copy(pjoin(datadir, "014790-NA12878.filt.no_gt_stats.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "014790-NA12878.filt.no_gt_stats_wgs.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "014790-NA12878.unfilt.no_gt_stats.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "014790-NA12878.unfilt.no_gt_stats_wgs.h5"), tmpdir)
    shutil.copy(pjoin(datadir, "014790-NA12878.chr1_head.vcf.gz"), tmpdir)
    with open(pjoin(tmpdir, "no_gt_report.config"), "w") as config_file:
        strg = f"""
[NOGTReport]
run_id = 014790-NA12878-UGAv3-205-CGATTCATGCTCGAT
pipeline_version = 1.1.0
filtered_h5_statistics = 014790-NA12878.filt.no_gt_stats.h5
filtered_h5_statistics_wgs = 014790-NA12878.filt.no_gt_stats_wgs.h5
h5_statistics = 014790-NA12878.unfilt.no_gt_stats.h5
h5_statistics_wgs = 014790-NA12878.unfilt.no_gt_stats_wgs.h5
annotation_intervals_names = LCR,EXOME,MAP_UNIQUE,LONG_HMER
filtered_vcf = 014790-NA12878.chr1_head.vcf.gz
interval_list = {test_dir}/resources/general/chr1_head/wgs_calling_regions.hg38.interval_list
ref_fasta = {test_dir}/resources/general/chr1_head/Homo_sapiens_assembly38.fasta
ref_fasta_dict = {test_dir}/resources/general/chr1_head/Homo_sapiens_assembly38.dict
is_somatic = false
h5_output = 014790-NA12878_no_gt_report.h5
"""
        config_file.write(strg)

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "report_wo_gt.ipynb",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "report_wo_gt.nbconvert.ipynb",
        "--template",
        "full",
        "--no-input",
        "--output",
        "no_gt_report.html",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
