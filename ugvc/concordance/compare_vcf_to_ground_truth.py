import argparse
import os.path

from ugvc.readwrite.bed_writer import BedWriter
from ugvc.readwrite.buffered_variant_reader import BufferedVariantReader
from ugvc import logger
from ugvc.sec.systematic_error_correction_call import SECCallType
from ugvc.utils.math import safe_divide
from ugvc.utils.pysam_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--relevant_coords', help='path to bed file describing relevant genomic subset to analyze', required=True)
    parser.add_argument('--ground_truth_vcf', required=True,
                        help='path to vcf.gz (tabix indexed) file containing true genotypes for this sample')
    parser.add_argument('--sample_id', help='id of analyzed sample', required=True)
    parser.add_argument(
        '--gvcf_file', help='path to gvcf file, (for getting the raw aligned reads information)', required=True)
    parser.add_argument(
        '--output_prefix', help='prefix to output files containing stats and info about errors', required=True)
    parser.add_argument(
        '--ignore_genotype', help='count as TP if have some variant in a positive locus', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if os.path.exists(args.gvcf_file):
        gvcf_reader = BufferedVariantReader(file_name=args.gvcf_file)
    else:
        logger.error('gvcf input does not exist')
        return

    ground_truth_reader = BufferedVariantReader(file_name=args.ground_truth_vcf)
    if args.sample_id not in ground_truth_reader.pysam_reader.header.samples:
        logger.error(f'sample {args.sample_id} not found in ground truth file {args.ground_truth_vcf}')
        return

    VCFComparator(args.relevant_coords,
                  ground_truth_reader,
                  gvcf_reader,
                  args.sample_id,
                  args.output_prefix,
                  args.ignore_genotype).assess()


class VCFComparator:

    def __init__(self,
                 relevant_coords_file: str,
                 ground_truth_reader: BufferedVariantReader,
                 gvcf_reader: BufferedVariantReader,
                 sample_id: str,
                 output_prefix: str,
                 ignore_genotype: bool):
        self.relevant_coords_file = relevant_coords_file
        self.ground_truth_reader = ground_truth_reader
        self.gvcf_reader = gvcf_reader
        self.sample_id = sample_id
        self.output_prefix = output_prefix
        self.ignore_genotype = ignore_genotype
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        self.log_stream = None
        self.positives = [0, 0]
        self.min_gq = 30

        # negatives := positions where observed genotype is interpreted as reference.
        #   read_negatives := negatives where observed genotype had an alternative (can be saved by SEC)
        #   unread_negatives := negatives where observed genotype exists, but didn't have an alternative allele
        #   uncovered_negatives := negatives where observed genotype does not exist
        self.read_negatives = [0, 0]
        self.unread_negatives = [0, 0]

        self.uncovered_negatives = [0, 0]
        self.novel = [0, 0]
        self.known = [0, 0]
        self.uncorrelated = [0, 0]
        self.unobserved = [0, 0]
        self.snp = [0, 0]
        self.indel = [0, 0]
        self.non_noise_allele = [0, 0]
        self.no_call_in_ground_truth = 0

    def assess(self):
        self.log_stream = open(f'{self.output_prefix}.log', 'w')
        fn_bed_writer = BedWriter(f'{self.output_prefix}.fn.bed')
        tp_bed_writer = BedWriter(f'{self.output_prefix}.tp.bed')
        fp_bed_writer =  BedWriter(f'{self.output_prefix}.fp.bed')
        uncorrelated_bed_writer = BedWriter(f'{self.output_prefix}.uncorrelated.bed')
        for line in open(self.relevant_coords_file):
            fields = line.split('\t')
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])  # 0-based coords
            for pos in range(int(start) + 1, int(end) + 1):
                ground_truth_variant = self.ground_truth_reader.get_variant(chrom, pos)
                observed_variant = self.gvcf_reader.get_variant(chrom, pos)

                if observed_variant is not None:
                    observed_sample_info = observed_variant.samples[0]
                    observed_genotype = get_genotype(observed_sample_info)
                    sorted_observed_genotype = '/'.join(sorted(observed_genotype.split('/')))
                    is_observed_variant = not is_ref_call(observed_sample_info) and observed_sample_info['GQ'] > self.min_gq
                else:
                    sorted_observed_genotype = observed_genotype = 'R'
                    is_observed_variant = False
                    observed_sample_info = None

                if ground_truth_variant is not None:
                    ground_truth_sample_info = ground_truth_variant.samples[self.sample_id]
                    ground_truth_genotype = get_genotype(ground_truth_sample_info)
                    # ignore positions with filtered variants in ground-truth
                    if not 'PASS' in ground_truth_variant.filter:
                        continue
                    sorted_ground_truth_genotype = '/'.join(sorted(ground_truth_genotype.split('/')))
                    is_ground_truth_variant = not is_ref_call(ground_truth_sample_info) and ground_truth_sample_info['GQ'] > self.min_gq
                    is_ground_truth_no_call = None in ground_truth_sample_info.alleles
                else:
                    sorted_ground_truth_genotype = ground_truth_genotype = 'R'  # assume reference genotype
                    is_ground_truth_variant = False
                    is_ground_truth_no_call = False

                # Skip uncorrelated loci
                if observed_variant is not None and 'ST' in observed_variant.format:
                    sec_type = observed_sample_info['ST']
                    if sec_type == SECCallType.uncorrelated.value:
                        if is_observed_variant or is_ground_truth_variant:
                            true_negative = not is_ground_truth_variant
                            if sec_type == SECCallType.uncorrelated.value:
                                self.uncorrelated[true_negative] += 1
                            var_type = 'snp' if is_snp(observed_variant.alleles) else 'indel'
                            alleles = get_alleles_str(observed_variant)
                            uncorrelated_bed_writer.write(chrom, pos - 1, pos, f'{true_negative}-{var_type}-{alleles}')
                            continue

                # Positive
                if is_observed_variant:
                    sec_type = observed_sample_info['ST']
                    # True positive
                    if sorted_observed_genotype == sorted_ground_truth_genotype or \
                            (self.ignore_genotype and is_ground_truth_variant):
                        if sec_type != SECCallType.non_noise_allele.value:
                            self.positives[True] += 1
                        self.__update_cross_section_counters(observed_genotype, observed_variant, match=True)
                        tp_bed_writer.write(chrom, pos - 1, pos, f'TP-{sec_type}-{observed_genotype}')
                    # False Positive
                    else:
                        if is_ground_truth_no_call:
                            fp_bed_writer.write(chrom, pos - 1, pos, f'no_call_ground_truth-{sec_type}-{observed_genotype}')
                            self.no_call_in_ground_truth += 1
                        else:
                            if sec_type != SECCallType.non_noise_allele.value:
                                self.positives[False] += 1
                            self.__update_cross_section_counters(observed_genotype, observed_variant, match=False)
                            self.__print(f'false {ground_truth_genotype}!={observed_genotype}\t{observed_variant}')
                            fp_bed_writer.write(chrom, pos - 1, pos,
                                                  f'FP-{sec_type}-{observed_genotype}!={ground_truth_genotype}')

                # Negative
                else:
                    # False Negative
                    if is_ground_truth_variant:
                        if observed_variant is None:
                            self.uncovered_negatives[False] += 1
                            fn_bed_writer.write(chrom, pos - 1, pos, f'uncovered-FN-{ground_truth_genotype}')
                        elif not has_candidate_alternatives(observed_variant):
                            fn_bed_writer.write(chrom, pos - 1, pos, f'unread-FN-{ground_truth_genotype}')
                            self.unread_negatives[False] += 1
                        else:
                            self.read_negatives[False] += 1
                            fn_bed_writer.write(chrom, pos - 1, pos, f'FN-{ground_truth_genotype}')
                            self.__print(f'missed {chrom}\t{pos}\t'
                                         f'{ground_truth_variant.ref}\t{",".join(ground_truth_variant.alts)}\t'
                                         f'{ground_truth_genotype}')
                    # True negative
                    else:
                        if observed_variant is None:
                            self.uncovered_negatives[True] += 1
                        elif not has_candidate_alternatives(observed_variant):
                            self.unread_negatives[True] += 1
                        else:
                            self.read_negatives[True] += 1

        fn_bed_writer.close()
        fp_bed_writer.close()
        tp_bed_writer.close()
        self.log_stream.close()
        self.__print_stats()

    def __print_stats(self):
        precision = safe_divide(self.positives[True], sum(self.positives))
        recall = safe_divide(self.positives[True], (self.positives[True] + self.read_negatives[False]))
        f1 = safe_divide(2 * precision * recall, precision + recall)

        def print_false_true(stream, name, category_tuple):
            stream.write(f'{name}\t{category_tuple[0]}\t{category_tuple[1]}\n')

        with open(f'{self.output_prefix}.stats.csv', 'w') as stats_stream:
            stats_stream.write(f'category\tFalse\tTrue\n')
            print_false_true(stats_stream, 'positives', self.positives)
            print_false_true(stats_stream, 'negatives', self.read_negatives)
            print_false_true(stats_stream, 'unread_negatives', self.unread_negatives)
            print_false_true(stats_stream, 'uncovered_negatives', self.uncovered_negatives)
            print_false_true(stats_stream, 'uncorrelated_negatives', self.uncorrelated)
            print_false_true(stats_stream, 'non_noise_allele', self.non_noise_allele)
            print_false_true(stats_stream, 'unobserved', self.unobserved)
            print_false_true(stats_stream, 'novel', self.novel)
            print_false_true(stats_stream, 'known', self.known)
            print_false_true(stats_stream, 'snp', self.snp)
            print_false_true(stats_stream, 'indel', self.indel)
            stats_stream.write(f'no_call_in_ground_truth\t{self.no_call_in_ground_truth}\n')

            stats_stream.write(f'precision\t{precision}\n')
            stats_stream.write(f'recall\t{recall}\n')
            stats_stream.write(f'f1\t{f1}\n')

    def __print(self, string: str):
        self.log_stream.write(f'{string}\n')

    def __update_cross_section_counters(self, observed_genotype, observed_variant, match: bool):
        non_noise_allele = False
        if 'ST' in observed_variant.format:
            sec_type = observed_variant.samples[0]['ST']
            if sec_type == SECCallType.novel.value:
                self.novel[match] += 1
            elif sec_type == SECCallType.known.value:
                self.known[match] += 1
            elif sec_type == SECCallType.non_noise_allele.value:
                self.non_noise_allele[match] += 1
                non_noise_allele = True
            elif sec_type == SECCallType.unobserved.value:
                self.unobserved[match] += 1

        if not non_noise_allele:
            if is_snp(observed_genotype.split('/')):
                self.snp[match] += 1
            else:
                self.indel[match] += 1


if __name__ == '__main__':
    main()
