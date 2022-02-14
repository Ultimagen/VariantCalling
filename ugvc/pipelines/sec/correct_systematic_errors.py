import argparse
import os.path
import pickle
import sys
from enum import Enum
from typing import List, TextIO

import pysam

from python.modules.filtering.blacklist import Blacklist
from python.pipelines.variant_filtering_utils import VariantSelectionFunctions
from ugvc import logger
from ugvc.readwrite.bed_writer import BedWriter
from ugvc.readwrite.buffered_variant_reader import BufferedVariantReader
from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from ugvc.sec.systematic_error_correction_call import SECCallType, SECCall
from ugvc.sec.systematic_error_correction_caller import SECCaller
from ugvc.sec.systematic_error_correction_record import SECRecord
from ugvc.utils.pysam_utils import get_genotype, get_filtered_alleles_str, \
    get_filtered_alleles_list


def get_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--relevant_coords', help='path to bed file describing relevant genomic subset to analyze',
                        required=True)
    parser.add_argument('--model', required=True,
                        help='path to pickle file containing conditional allele distributions per position of interest')
    parser.add_argument('--gvcf', required=True,
                        help='path to gvcf file, (for getting the raw aligned reads information)')
    parser.add_argument('--output_file', help='path to output file (vcf/vcf.gz/bed/pickle)')
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
    parser.add_argument('--filter_uncorrelated', default=False, action='store_true',
                        help='filter variants in positions where ref and alt conditioned genotype have similar distributions')
    args = parser.parse_args(argv)
    return args


class SystematicErrorCorrector:
    """
    Given an observed gvcf file, and expected-distributions model, decide for each variant,
    if it should be kept (observation is significantly different from the expected systematic error)
    Or should be filtered (observation is similar to expected systematic error)

    Works in two modes
    1. Regular - use all available information from the expected distributions model
    2. Novel detection only - do not use information on expected distributions given known alternative genotypes.

    Sites where the model is ambiguous, (observation can be explained equally well by two different ground-truth genotypes)
    are identified as uncorrelated sites.
    The default behaviour is to keep such variants as they are, since the method has no info about them.


    --filter_uncorrelated will cause calling these sites as reference.
        Notice this will filter out only sites where one of the ground-truth genotypes is the reference genotype
        Since if both alternatives are non-ref, than wither way the variant is probably real.
    """

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
                 replace_to_known_genotype: bool,
                 filter_uncorrelated: bool):
        self.relevant_coords = relevant_coords
        self.distributions_per_chromosome = conditional_allele_distributions.distributions_per_chromosome
        self.gvcf_reader = gvcf_reader
        self.output_file = output_file
        self.filter_uncorrelated = filter_uncorrelated

        self.output_type = None
        if output_file.endswith('.bed'):
            self.output_type = OutputType.bed
        elif output_file.endswith('.vcf') or output_file.endswith('.vcf.gz'):
            self.output_type = OutputType.vcf
        elif output_file.endswith('.pickle'):
            self.output_type = OutputType.pickle
        else:
            raise ValueError('output file must end with bed/vcf/vcf.gz/pickle suffixes')

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
        self.gvcf_reader.header.add_meta('FORMAT', items=[('ID', "SPV"),
                                                          ('Number', 1),
                                                          ('Type', 'Float'),
                                                          ('Description', 'SECPValue')])

        # Initialize output writers
        vcf_writer = None
        bed_writer = None
        if self.output_type == OutputType.vcf:
            vcf_writer = pysam.libcbcf.VariantFile(self.output_file, mode='w', header=self.gvcf_reader.header)
        if self.output_type == OutputType.bed:
            bed_writer = BedWriter(self.output_file)

        # Initialize (chr, pos) tuples, relevant only for pickle output
        chr_pos_tuples = []

        log_stream = open(f'{self.output_file}.log', 'w')

        # For each position in the relevant-coords bed file, compare observation to expected distribution of alleles
        for line in self.relevant_coords:
            fields = line.split('\t')
            chrom, start, end = fields[0], fields[1], fields[2]

            for pos in range(int(start) + 1, int(end) + 1):
                observed_variant = self.gvcf_reader.get_variant(chrom, pos)

                # skip position if no variant was observed
                if observed_variant is None:
                    continue

                sample_info = observed_variant.samples[0]

                # Handle no-call
                if None in sample_info['GT']:

                    if self.output_type == OutputType.vcf:
                        vcf_writer.write(observed_variant)
                    continue

                # Check if called alternative allele matches excluded (noise) allele
                called_non_excluded_alt = False
                if len(fields) > 3:
                    excluded_alleles = fields[3].replace('[', '').replace(']', '').replace("'", "").strip().split(',')
                    called_alts = set(get_filtered_alleles_list(sample_info)).intersection(observed_variant.alts)
                    called_non_excluded_alt = len(called_alts.difference(excluded_alleles)) > 0

                # Call a non_noise_allele in case alternative allele is not excluded
                # e.g a SNP in a position where the noise in a hmer indel
                if called_non_excluded_alt:
                    observed_alleles = get_filtered_alleles_str(observed_variant)
                    call = SECCall(SECCallType.non_noise_allele, observed_alleles, get_genotype(sample_info), [])
                # Execute SEC caller on the observed variant and expected_distribution loaded from the memDB (pickle)
                else:
                    distributions_per_pos = self.distributions_per_chromosome.get(chrom, {})
                    expected_distribution = distributions_per_pos.get(pos, None)
                    call = self.caller.call(observed_variant, expected_distribution)

                for sec_record in call.sec_records:
                    log_stream.write(f'{sec_record}\n')


                if self.output_type == OutputType.pickle:
                    self.__process_call_pickle_output(call, chr_pos_tuples, chrom, pos)

                if self.output_type == OutputType.bed:
                    self.__process_call_bed_output(bed_writer, call, chrom, fields, pos)

                if self.output_type == OutputType.vcf:
                    self.__process_call_vcf_output(call, observed_variant, sample_info, vcf_writer)

                log_stream.write(f'{chrom}\t{pos}\t{call}\n\n')

        if self.output_type == OutputType.vcf:
            vcf_writer.close()
            pysam.tabix_index(self.output_file, preset='vcf', force=True)
        if self.output_type == OutputType.bed:
            bed_writer.close()
        if self.output_type == OutputType.pickle:
            exclusion_filter = Blacklist(blacklist=set(chr_pos_tuples),
                                         annotation='SEC',
                                         selection_fcn=VariantSelectionFunctions.ALL,
                                         description='Variant is filtered since alt alleles have high likelihood to be '
                                                     'generated by a systematic error')
            with open(self.output_file, 'wb') as out_pickle_file:
                pickle.dump(exclusion_filter, out_pickle_file)
        log_stream.close()

    def __process_call_vcf_output(self, call, observed_variant, sample_info, vcf_writer):
        """
        Write all vcf lines with additional:
        1. ST field: (reference, novel, known, uncorrelated, non_noise_allele)
        2. updated GQ whenever appropriate (known variants)
        3. Corrected genotype, to ref genotype when filtered the variant
        """
        sample_info['ST'] = str(call.call_type.value)
        sample_info['SPV'] = call.novel_variant_p_value
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

    def __process_call_bed_output(self, bed_writer, call, chrom, fields, pos):
        """
        output only positions which were decided to have the reference genotype
        (or uncorrelated if directed to filter them)
        """

        if call.call_type == SECCallType.reference or (
                call.call_type == SECCallType.uncorrelated and self.filter_uncorrelated):
            if len(fields) > 3:
                bl_alleles = fields[3].strip()
                bed_writer.write(chrom, pos - 1, pos, bl_alleles)
            else:
                bed_writer.write(chrom, pos - 1, pos, call.call_type.value)

    def __process_call_pickle_output(self, call, chr_pos_tuples, chrom, pos):
        """
        output only positions which were decided to have the reference genotype
        (or uncorrelated if directed to filter them)
        """
        if call.call_type == SECCallType.reference or \
                (call.call_type == SECCallType.uncorrelated and self.filter_uncorrelated):
            chr_pos_tuples.append((chrom, pos))


class OutputType(Enum):
    vcf = 1
    bed = 2
    pickle = 3


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
                             replace_to_known_genotype=args.replace_to_known_genotype,
                             filter_uncorrelated=args.filter_uncorrelated).correct_systematic_errors()
    relevant_coords.close()


if __name__ == '__main__':
    main(sys.argv[1:])
