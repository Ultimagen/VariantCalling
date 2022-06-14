from enum import Enum

COVERAGE = "coverage"
GCS_OAUTH_TOKEN = "GCS_OAUTH_TOKEN"


class FileExtension(Enum):
    """File Extension enum"""

    PARQUET = ".parquet"
    HDF = ".hdf"
    H5 = ".h5"
    CSV = ".csv"
    TSV = ".tsv"
    BAM = ".bam"
    CRAM = ".cram"
    PNG = ".png"
    JPEG = ".jpeg"
    FASTA = ".fasta"
    TXT = ".txt"
