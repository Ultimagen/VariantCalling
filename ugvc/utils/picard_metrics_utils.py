import pandas as pd

def parse_cvg_metrics(metric_file):
    """Parses Picard WGScoverage metrics file

    Parameters
    ----------
    metric_file : str
        Picard metric file

    Returns
    -------
    res1 : str
        Picard file metrics class
    res2 : pd.DataFrame
        Picard metrics table
    res3 : pd.DataFrame
        Picard Histogram output
    """
    with open(metric_file, encoding="ascii") as infile:
        out = next(infile)
        while not out.startswith("## METRICS CLASS"):
            out = next(infile)

        res1 = out.strip().split("\t")[1].split(".")[-1]
        res2 = pd.read_csv(infile, sep="\t", nrows=1)
    try:
        with open(metric_file, encoding="ascii") as infile:
            out = next(infile)
            while not out.startswith("## HISTOGRAM\tjava.lang.Integer"):
                out = next(infile)

            res3 = pd.read_csv(infile, sep="\t")
    except StopIteration:
        res3 = None
    return res1, res2, res3