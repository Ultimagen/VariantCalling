import pandas as pd
BED_COLUMN_CHROM = "chrom"
BED_COLUMN_CHROM_START = "chromStart"
BED_COLUMN_CHROM_END = "chromEnd"


class BedWriter:
    def __init__(self, output_file: str):
        self.fh = open(output_file, "w")

    def write(
        self,
        chrom: str,
        start: int,
        end: int,
        description: str = None,
        score: float = None,
    ):
        if start > end:
            raise ValueError(f"start > end in write bed file: {start} > {end}")
        self.fh.write(f"{chrom}\t{start}\t{end}")
        if description is not None:
            self.fh.write(f"\t{description}")
        if score is not None:
            self.fh.write(f"\t{score}")
        self.fh.write("\n")

    def close(self):
        self.fh.close()


def parse_intervals_file(intervalfile: str, threshold: int = 0) -> pd.DataFrame:
    """Parses bed file

    Parameters
    ----------
    intervalfile : str
        Input BED file
    threshold : int, optional
        minimal length of interval to output (default = 0)

    Returns
    -------
    pd.DataFrame
        Output dataframe with columns chromosome, start, end
    """
    df = pd.read_csv(
        intervalfile,
        names=["chromosome", "start", "end"],
        usecols=[0, 1, 2],
        index_col=None,
        sep="\t",
    )
    if threshold > 0:
        df = df[df["end"] - df["start"] > threshold]
    df.sort_values(["chromosome", "start"], inplace=True)
    return df