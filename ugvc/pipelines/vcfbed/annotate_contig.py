import argparse
import pickle

from ugvc.vcfbed.variant_annotation import VcfAnnotator

# Helper script to annotate a single contig of a VCF file


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf_in", required=True, help="path to input vcf")
    parser.add_argument("--vcf_out", required=True, help="path to otuput vcf")
    parser.add_argument(
        "--annotators_pickle",
        required=True,
        help="path to list of VcfAnnotator pickled classes to annotate with",
    )
    parser.add_argument(
        "--contig",
        required=True,
        help="name of contig to process (this script analyses a contig at a time, "
        "use VcfAnnotator.process_vcf to process multiple contigs)",
    )
    parser.add_argument("--chunk_size", required=True, help="size of chunks to process in memory")
    return parser.parse_args(argv)


def run(argv):
    """Helper script to annotate a single contig of a VCF file"""
    args = get_args(argv[1:])
    with open(args.annotators_pickle, "rb") as f:
        annotators = pickle.load(f)
    VcfAnnotator.process_contig(
        vcf_in=args.vcf_in,
        vcf_out=args.vcf_out,
        annotators=annotators,
        contig=args.contig,
        chunk_size=args.chunk_size,
    )
