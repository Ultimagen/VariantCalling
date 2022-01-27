import argparse
import os.path
import pickle
import sys
from typing import List, TextIO

import pysam

from ugvc import logger
from ugvc.readwrite.bed_writer import BedWriter
from ugvc.readwrite.buffered_variant_reader import BufferedVariantReader
from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from ugvc.sec.systematic_error_correction_call import SECCallType, SECCall
from ugvc.sec.systematic_error_correction_caller import SECCaller
from ugvc.sec.systematic_error_correction_record import SECRecord
from ugvc.utils.pysam_utils import fix_ultima_info, get_genotype, get_filtered_alleles_str, \
    get_filtered_alleles_list


def get_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--relevant_coords', help='path to bed file describing relevant genomic subset to analyze',
                        required=True)
    parser.add_argument('--model', required=True,
                        help='path to pickle file containing conditional allele distributions per position of interest')
    parser.add_argument('--gvcf', required=True,
                        help='path to gvcf file, (for getting the raw aligned reads information)')
    parser.add_argument('--output_file', help='path to output file (vcf/vcf.gz/bed)')
    parser.add_argument('--strand_enrichment_pval_thresh', default=0.000001, type=float,
                        help='p-value threshold for strand enrichment of alt-alleles (forward_pval * reverse_pval)')
    parser.add_argument('--lesser_strand_enrichment_pval_thresh', default=0.001, type=float,
                        help='p-value threshold for the strand with higher p-value max(forward_pval, reverse_pval)')
    parser.add_argument('--min_gt_correlation', default=0.99, type=float,
                        help="don't call loci with lower correlation between ground-truth and observed genotypes")
    parser.add_argument('--noise_ratio_for_unobserved_snps', default=0.02, type=float,
                        help='expected probability of seeing alternative in unobserved snp for noisy positions')
    parser.add_argument('--noise_ratio_for_unobserved_indels', default=0.05, type=float,
                        help='expected probability of seeing alternative in unobserved indel for noisy positions')
    parser.add_argument('--novel_detection_only', default=False, action='store_true',
                        help='do not use information on known variants')
    parser.add_argument('--replace_to_known_genotype', default=False, action='store_true',
                        help='in case reads match known genotype from ground-truth, use this genotype')
    args = parser.parse_args(argv)
    return args


class SystematicErrorCorrector:

    def __init__(self,
                 relevant_coords: TextIO,
                 conditional_allele_distributions: ConditionalAlleleDistributions,
                 gvcf_reader: BufferedVariantReader,
                 strand_enrichment_pval_thresh: float,
                 lesser_strand_enrichment_pval_thresh: float,
                 min_gt_correlation: float,
                 noise_ratio_for_unobserved_snps: float,
                 noise_ratio_for_unobserved_indels: float,
                 output_file: str,
                 novel_detection_only: bool,
                 replace_to_known_genotype: bool):
        self.relevant_coords = relevant_coords
        self.distributions_per_chromosome = conditional_allele_distributions.distributions_per_chromosome
        self.gvcf_reader = gvcf_reader
        self.output_file = output_file
        if output_file.endswith('.bed'):
            self.output_bed = True
            self.output_vcf = False
        elif output_file.endswith('.vcf') or output_file.endswith('.vcf.gz'):
            self.output_vcf = True
            self.output_bed = False
        else:
            raise ValueError('output file must end with bed or vcf or vcf.gz suffixes')

        self.caller = SECCaller(strand_enrichment_pval_thresh,
                                lesser_strand_enrichment_pval_thresh,
                                min_gt_correlation,
                                novel_detection_only,
                                replace_to_known_genotype)
        SECRecord.noise_ratio_for_unobserved_snps = noise_ratio_for_unobserved_snps
        SECRecord.noise_ratio_for_unobserved_indels = noise_ratio_for_unobserved_indels

    def correct_systematic_errors(self):
        self.gvcf_reader.header.add_meta('FORMAT', items=[('ID', "ST"),
                                                          ('Number', 1),
                                                          ('Type', 'String'),
                                                          ('Description', 'SECType')])

        vcf_writer = None
        bed_writer = None
        if self.output_vcf:
            vcf_writer = pysam.libcbcf.VariantFile(self.output_file, mode='w', header=self.gvcf_reader.header)
        if self.output_bed:
            bed_writer = BedWriter(self.output_file)

        log_stream = open(f'{self.output_file}.log', 'w')
        for line in self.relevant_coords:
            fields = line.split('\t')
            chrom, start, end = fields[0], fields[1], fields[2]


            for pos in range(int(start) + 1, int(end) + 1):
                observed_variant = self.gvcf_reader.get_variant(chrom, pos)

                if observed_variant is None:
                    continue

                sample_info = observed_variant.samples[0]
                # Handle no-call
                if None in sample_info['GT']:
                    if self.output_vcf:
                        fix_ultima_info(observed_variant, self.gvcf_reader.header)
                        vcf_writer.write(observed_variant)
                    continue

                called_non_excluded_alt = False
                if len(fields) > 3:
                    excluded_alleles = fields[3].replace('[', '').replace(']', '').replace("'", "").strip().split(',')
                    called_alts = set(get_filtered_alleles_list(sample_info)).intersection(observed_variant.alts)
                    called_non_excluded_alt = len(called_alts.difference(excluded_alleles)) > 0

                if called_non_excluded_alt:
                    observed_alleles = get_filtered_alleles_str(observed_variant)
                    call = SECCall(SECCallType.non_noise_allele, observed_alleles, get_genotype(sample_info), [])
                else:
                    distributions_per_pos = self.distributions_per_chromosome.get(chrom, {})
                    expected_distribution = distributions_per_pos.get(pos, None)
                    call = self.caller.call(observed_variant, expected_distribution)

                for sec_record in call.sec_records:
                    log_stream.write(f'{sec_record}\n')

                if self.output_bed:
                    # output only positions which were decided to have the reference genotype (or uncorrelated, or non-noise allele)
                    if call.call_type == SECCallType.reference or call.call_type == SECCallType.uncorrelated \
                            or call.call_type == SECCallType.non_noise_allele:
                        if len(fields) > 3:
                            bl_alleles = fields[3].strip()
                            bed_writer.write(chrom, pos - 1, pos, bl_alleles)
                        else:
                            bed_writer.write(chrom, pos - 1, pos, call.call_type.value)
                if self.output_vcf:
                    fix_ultima_info(observed_variant, self.gvcf_reader.header)

                    sample_info['ST'] = str(call.call_type.value)

                    if call.call_type == SECCallType.reference:
                        gt = call.get_genotype_indices_tuple()
                        sample_info['GT'] = gt
                    elif call.call_type == SECCallType.known:
                        observed_variant.alleles = call.get_alleles_list()
                        sample_info = observed_variant.samples[0]
                        sample_info['GT'] = call.get_genotype_indices_tuple()

                    if call.genotype_quality is not None:
                        sample_info['GQ'] = call.genotype_quality
                    vcf_writer.write(observed_variant)

                log_stream.write(f'{chrom}\t{pos}\t{call}\n\n')

        if self.output_vcf:
            vcf_writer.close()
            pysam.tabix_index(self.output_file, preset='vcf', force=True)
        if self.output_bed:
            bed_writer.close()
        log_stream.close()

def main(argv: List[str]):
    """
    filter out variants which appear like systematic-errors, while keeping those which are not well explained by errors
    """
    args = get_args(argv)
    if os.path.exists(args.gvcf):
        gvcf_reader = BufferedVariantReader(args.gvcf)
    else:
        logger.error('gvcf input does not exist')
        return

    relevant_coords = open(args.relevant_coords, 'r')

    with open(args.model, 'rb') as fh:
        conditional_allele_distributions: ConditionalAlleleDistributions = pickle.load(fh)

    SystematicErrorCorrector(relevant_coords=relevant_coords,
                             conditional_allele_distributions=conditional_allele_distributions,
                             gvcf_reader=gvcf_reader,
                             strand_enrichment_pval_thresh=args.strand_enrichment_pval_thresh,
                             lesser_strand_enrichment_pval_thresh=args.lesser_strand_enrichment_pval_thresh,
                             min_gt_correlation=args.min_gt_correlation,
                             noise_ratio_for_unobserved_snps=args.noise_ratio_for_unobserved_snps,
                             noise_ratio_for_unobserved_indels=args.noise_ratio_for_unobserved_indels,
                             output_file=args.output_file,
                             novel_detection_only=args.novel_detection_only,
                             replace_to_known_genotype=args.replace_to_known_genotype).correct_systematic_errors()
    relevant_coords.close()


if __name__ == '__main__':
    main(sys.argv[1:])
