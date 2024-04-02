import sys

import pysam
from simppl.cli import get_parser

from ugvc import logger


def is_homopolymer_snp(record, reference, min_query_hmer_size, min_initial_qual, min_distance_from_edge):
    # Get the position of the variant
    variant_pos = record.pos
    contig = record.chrom

    # Check if the variant is a SNP
    if (
        len(record.ref) == 1
        and len(record.alts) == 1
        and len(record.alts[0]) == 1
        and "PASS" not in record.filter.keys()
        and record.qual >= min_initial_qual
    ):
        # Get the base at the variant position in the reference
        alt_base = record.alts[0]

        hmer_size = 1  # consider alt base as initial hmer of size 1

        # count how many of the surrounding bases are the same as the alt base
        reference_seq = reference.fetch(
            contig, variant_pos - min_query_hmer_size - 1, variant_pos + min_query_hmer_size
        )
        upstream_hmer_length = 0
        downstream_hmer_length = 0
        base_before_hmer = ""
        base_after_hmer = ""
        for base in reference_seq[min_query_hmer_size + 1 :]:
            if base == alt_base:
                hmer_size += 1
                downstream_hmer_length += 1
            else:
                base_after_hmer = base
                break

        for base in reference_seq[min_query_hmer_size - 1 :: -1]:
            if base == alt_base:
                hmer_size += 1
                upstream_hmer_length += 1
            else:
                base_before_hmer = base
                break

        is_tandem_repeat = (
            base_before_hmer == base_after_hmer
            and base_before_hmer == record.ref
            and upstream_hmer_length == downstream_hmer_length
        )
        if (
            hmer_size >= min_query_hmer_size
            and not is_tandem_repeat
            and min(upstream_hmer_length, downstream_hmer_length) >= min_distance_from_edge
        ):
            info = f"refseq : {reference_seq} hmer_size: {hmer_size}"
            return True, info

        return False, f"hmer_size: {hmer_size} is not above min_query_hmer_size or is a tandem repeat"

    return False, "not a filtered snp above min_qual"


def init_parser():
    parser = get_parser("calibrate_bridging_snvs", run.__doc__)
    parser.add_argument("--vcf", required=True, help="Path to the VCF file")
    parser.add_argument("--reference", required=True, help="Path to the reference genome")
    parser.add_argument("--output", required=True, help="name of output vcf file")
    parser.add_argument(
        "--min_query_hmer_size",
        default=5,
        type=int,
        help="min size of the homopolymer in the query genome (with SNV alt allele) to be considered",
    )
    parser.add_argument("--min_initial_qual", default=5, type=int, help="min quality of the initial SNV call")
    parser.add_argument("--min_tumor_vaf", default=0.2, type=float, help="min variant allele frequency in the tumor")
    parser.add_argument("--max_normal_vaf", default=0.1, type=float, help="max variant allele frequency in the normal")
    parser.add_argument("--min_normal_depth", default=10, type=int, help="min depth in the normal")
    parser.add_argument(
        "--min_distance_from_edge", default=0, type=int, help="min distance from the edge of the homopolymer"
    )
    parser.add_argument("--set_qual", default=20, type=int, help="set the quality of the SNV to this value")
    return parser


def run(argv):
    """
    Un-filter SNVs which generate a long homopolymer, have borderline quality
    and have a high VAF in the tumor and low VAF in the normal
    * DV often filters such true SNVs due to low confidence of the allele (SNV / deletion)
    """

    args = init_parser().parse_args(argv[1:])

    # Open the VCF file
    vcf = pysam.VariantFile(args.vcf)

    # Open the reference genome
    reference = pysam.FastaFile(args.reference)

    # Create a new VCF file for output
    out_vcf = pysam.VariantFile(args.output, "w", header=vcf.header)

    # Iterate over the records in the VCF file
    for record in vcf:
        is_hm_snp, info = is_homopolymer_snp(
            record, reference, args.min_query_hmer_size, args.min_initial_qual, args.min_distance_from_edge
        )
        if is_hm_snp:
            normal_depth = record.samples[0]["BG_DP"]
            tumor_vaf = sum(record.samples[0]["AD"][1:]) / record.samples[0]["DP"]
            normal_vaf = sum(record.samples[0]["BG_AD"][1:]) / max(0.01, normal_depth)
            if (
                tumor_vaf >= args.min_tumor_vaf
                and normal_vaf <= args.max_normal_vaf
                and normal_depth > args.min_normal_depth
            ):
                logger.info(info)
                logger.info(str(record))
                record.filter.add("PASS")
                record.qual = args.set_qual

        out_vcf.write(record)
    out_vcf.close()
    pysam.tabix_index(args.output, preset="vcf")


if __name__ == "__main__":
    run(sys.argv)
