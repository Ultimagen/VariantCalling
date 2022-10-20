from __future__ import annotations
import argparse
import pysam
from pysam import VariantRecord
import math
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser('convert_haploid_regions', run.__doc__)
    parser.add_argument('--input_vcf', help='path to vcf file', required=True)
    parser.add_argument('--output_vcf', required=True)
    parser.add_argument('--haploid_regions', required=True,
                        help='path to bed file defining haploid regions.'
                             'Give hg38_non_par to use hardcoded hg38 non-par regions')

    return parser.parse_args(argv[1:])


def load_regions_from_bed(bed_file: str) -> list[tuple[str, int, int]]:
    string_fields = [line.split('\t') for line in open(bed_file).readlines()]
    regions = [(fields[0], int(fields[1]), int(fields[2])) for fields in string_fields]
    return regions


def in_regions(chrom: str, pos: int, regions: list[tuple[str, int, int]]):
    """
    inefficient implementation for looking if a position is contained in a list of genomic regions
    Do not use for large regions list
    """
    for region in regions:
        (region_chrom, region_start, region_end) = region
        if chrom == region_chrom and (region_start < pos <= region_end):
            return True
    return False


def convert_to_haploid(variant: VariantRecord):
    call = variant.samples[0]
    pls = call['PL']
    num_alleles = len(variant.alts) + 1
    # already haploid
    if len(pls) == 2:
        return variant
    else:
        un_normalized_probs = [10 ** (pl / -10) for pl in pls]
        homozygous_un_normalized_probs = []
        for i in range(num_alleles):
            for j in range(i, num_alleles):
                if i == j:
                    pl_index = int(i * (i + 1) / 2 + j)
                    homozygous_un_normalized_probs.append(un_normalized_probs[pl_index])
        haploid_probs = tuple([p / sum(homozygous_un_normalized_probs) for p in homozygous_un_normalized_probs])
        haploid_pls = [int(-10 * math.log10(p)) for p in haploid_probs]
        min_pl = min(haploid_pls)
        haploid_pls = [pl - min_pl for pl in haploid_pls]

        gq = 10000
        for i, pl in enumerate(haploid_pls):
            if pl == 0:
                called_haplotype = i
            elif pl < gq:
                gq = pl
        # maintain no call
        if call['GT'][0] is None:
            called_haplotype = None
        call['GT'] = called_haplotype
        call['GQ'] = gq
        call['PL'] = haploid_pls
    return variant


def run(argv):
    """
    Convert genotypes of specified regions to haploid calls, maintaining correct GT,GQ,PL
    """
    args = parse_args(argv)

    if args.haploid_regions == 'hg38_non_par':
        #      _________
        #      \       /
        #       \     /      (chrX with par regions depicted as =, haploid regions as -)
        #    -==--   --==-
        #    -==-------==    (chrY with par regions depicted as =, haploid regions as -)
        haploid_regions = [('chrX', 1, 10001),
                           ('chrX', 2781479, 155701383),
                           ('chrX', 156030895, 156040895),
                           ('chrY', 1, 10001),
                           ('chrY', 2781479, 56887903)]
    else:
        haploid_regions = load_regions_from_bed(args.haploid_regions)

    vcf_reader = pysam.VariantFile(args.input_vcf)
    vcf_writer = pysam.libcbcf.VariantFile(args.output_vcf, mode="w", header=vcf_reader.header)
    for variant in vcf_reader:
        if in_regions(variant.chrom, variant.pos, haploid_regions):
            vcf_writer.write(convert_to_haploid(variant))
        else:
            vcf_writer.write(variant)


if __name__ == '__main__':
    run(sys.argv)
