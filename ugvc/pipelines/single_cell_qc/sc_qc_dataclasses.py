from dataclasses import dataclass
import os
from typing import List


@dataclass
class Inputs:
    trimmer_stats_csv: str
    trimmer_histogram_csv: List[str]
    trimmer_failure_codes_csv: str
    sorter_stats_csv: str
    star_stats: str
    star_reads_per_gene: str
    r2_subsample: str

    def __post_init__(self):
        for _, value in self.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    assert os.path.isfile(item), f"{item} not found"
            else:
                assert os.path.isfile(value), f"{value} not found"


@dataclass
class Thresholds:
    pass_trim_rate: float  # minimal %trimmed
    read_length: int  # expected read length
    fraction_below_read_length: float  # fraction of reads below read length
    percent_aligned: float  # minimal % of reads aligned