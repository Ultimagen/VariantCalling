import argparse

from simppl.simple_pipeline import SimplePipeline

from ugvc.vcfbed.filter_bed import intersect_bed_regions


def get_parser():
    parser = argparse.ArgumentParser(prog="intersect_bed_regions.py", description=run.__doc__)
    parser.add_argument(
        "--include-regions",
        nargs="+",
        required=True,
        help="List of paths to BED files to be intersected.",
    )
    parser.add_argument(
        "--exclude-regions",
        nargs="*",
        default=None,
        help="List of paths to BED or VCF files to be subtracted from the intersected result.",
    )
    parser.add_argument("--output-bed", default="output.bed", help="Path to the output BED file.")
    parser.add_argument(
        "--assume-input-sorted",
        action="store_true",
        help="If set, assume that the input files are already sorted. "
        "If not set, the function will sort them on-the-fly.",
    )
    parser.add_argument(
        "--max_mem",
        type=int,
        default=None,
        help="Maximum memory in bytes allocated for the sort-bed operations. If not specified, "
        "the function will allocate 80 percent of the available system memory.",
    )

    return parser


def run(argv):
    """Intersect BED regions with the option to subtract exclude regions."""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    sp = SimplePipeline(args.fc, args.lc, debug=args.d, print_timing=True)
    intersect_bed_regions(
        include_regions=args.include_regions,
        exclude_regions=args.exclude_regions,
        output_bed=args.output_bed,
        assume_input_sorted=args.assume_input_sorted,
        max_mem=args.max_mem,
        sp=sp,
    )
