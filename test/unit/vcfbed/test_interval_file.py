import os
from os.path import exists
from os.path import join as pjoin
from test import get_resource_dir, test_dir

from simppl.simple_pipeline import SimplePipeline

from ugvc.vcfbed.interval_file import IntervalFile

inputs_dir = get_resource_dir(__file__)
common_dir = pjoin(test_dir, "resources", "general")


def test_interval_file_init_bed_input():
    bed1 = pjoin(inputs_dir, "bed1.bed")
    ref_genome = pjoin(common_dir, "sample.fasta")
    interval_list_path = pjoin(inputs_dir, "bed1.interval_list")

    sp = SimplePipeline(0, 100, False)
    interval_file = IntervalFile(sp, bed1, ref_genome, None)

    assert interval_file.as_bed_file() == bed1
    assert interval_file.as_interval_list_file() == interval_list_path
    assert exists(interval_list_path)
    assert not interval_file.is_none()
    os.remove(interval_list_path)


def test_interval_file_init_interval_list_input(mocker):
    interval_list = pjoin(inputs_dir, "interval_list1.interval_list")
    ref_genome = pjoin(common_dir, "sample.fasta")
    bed_path = pjoin(inputs_dir, "interval_list1.bed")

    sp = SimplePipeline(0, 100, False)
    interval_file = IntervalFile(sp, interval_list, ref_genome, None)

    assert interval_file.as_bed_file() == bed_path
    assert interval_file.as_interval_list_file() == interval_list
    assert exists(bed_path)
    assert not interval_file.is_none()
    os.remove(bed_path)


def test_interval_file_init_error():
    ref_genome = pjoin(common_dir, "sample.fasta")
    sp = SimplePipeline(0, 100, False)
    interval_file = IntervalFile(sp, ref_genome, ref_genome, None)
    assert interval_file.as_bed_file() is None
    assert interval_file.as_interval_list_file() is None
    assert interval_file.is_none()
