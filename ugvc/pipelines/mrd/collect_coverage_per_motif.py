#!/env/python
import argparse
from typing import List

from ugvc.mrd.coverage_utils import collect_coverage_per_motif


def __parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_coverage_per_motif", description=run.__doc__
    )
    parser.add_argument(
        "input", nargs="+", type=str, help="input depth files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="""Path to which output dataframe will be written in hdf format""",
    )
    parser.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="reference fasta, only required for motif annotation,"
        " most likely gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta"
        " but it must be localized",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=4,
        help="Process 1 in every N positions - makes the code run faster the larger N is (default 100). "
        "In the output dataframe df, df['coverage'] = df['count'] * N",
    )
    parser.add_argument(
        "-m",
        "--motif_length",
        type=int,
        default=4,
        help="Maximal motif length to collect coverage for",
    )
    parser.add_argument(
        "--show_stats",
        default=False,
        action="store_true",
        help="Print motif statistics",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel (default -1 for max)",
    )
    return parser.parse_args(argv)


def run(argv: List[str]):
    """Collect coverage per motif from a collection of depth files"""
    args_in = __parse_args(argv)
    collect_coverage_per_motif(
        depth_files=args_in.input,
        reference_fasta=args_in.reference_fasta,
        outfile=args_in.output,
        show_stats=args_in.show_stats,
        n_jobs=args_in.jobs,
        size=args_in.motif_length,
        N=args_in.N,
    )
