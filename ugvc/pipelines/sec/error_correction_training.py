#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Correct statistics for systematic error correction from gVCFs
# CHANGELOG in reverse chronological order


import argparse
import os.path

import pysam

from ugvc import logger
from ugvc.vcfbed.buffered_variant_reader import BufferedVariantReader
from ugvc.sec.allele_counter import count_alleles_in_gvcf, count_alleles_in_pileup
from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.sec.efm_factory import pileup_to_efm
from ugvc.vcfbed.pysam_utils import (
    get_alleles_str,
    get_filtered_alleles_str,
    get_genotype_indices,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relevant_coords",
        help="path to bed file describing relevant genomic subset to analyze",
        required=True,
    )

    parser.add_argument(
        "--ground_truth_vcf",
        required=True,
        help="path to vcf.gz (tabix indexed) file containing true genotypes for this sample",
    )
    parser.add_argument("--sample_id", help="id of analyzed sample", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--sam_file",
        help="path to sam/bam/cram file  (for getting processed aligned reads information)",
    )
    parser.add_argument(
        "--gvcf_file",
        help="path to gvcf file, (for getting the raw aligned reads information)",
    )

    args = parser.parse_args()
    return args


def main():
    """
    sam_file processing is efficient only if relevant_coords cover a small sub-set of genomic positions
    """
    args = get_args()

    if args.sam_file:
        sam_reader = pysam.AlignmentFile(args.sam_file, "rb")
        gvcf_reader = None
    elif args.gvcf_file:
        if os.path.exists(args.gvcf_file):
            gvcf_reader = BufferedVariantReader(file_name=args.gvcf_file)
        else:
            logger.error("gvcf input does not exist")
            return

        sam_reader = None
    else:
        raise ValueError("Must define either sam_file or gvcf_file input")

    ground_truth_reader = BufferedVariantReader(file_name=args.ground_truth_vcf)
    if args.sample_id not in ground_truth_reader.pysam_reader.header.samples:
        logger.error(
            f"sample {args.sample_id} not found in ground truth file {args.ground_truth_vcf}"
        )
        return

    with open(args.output_file, "w") as out_stream:
        error_correction_training(
            args.relevant_coords,
            ground_truth_reader,
            gvcf_reader,
            sam_reader,
            args.sample_id,
            out_stream,
        )


def error_correction_training(
    relevant_coords_file: str,
    ground_truth_reader: BufferedVariantReader,
    gvcf_reader: BufferedVariantReader,
    sam_reader: pysam.AlignmentFile,
    sample_id: str,
    out_stream,
):
    for line in open(relevant_coords_file):
        fields = line.split("\t")
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])  # 0-based coords
        for pos in range(start + 1, end + 1):
            ground_truth_variant = ground_truth_reader.get_variant(chrom, pos)

            # sam input
            if sam_reader is not None:
                # only way to access a pileup column is through iteration,
                # NOTICE: pileup will iterate ALL positions with reads covering pos
                for pileup_column in sam_reader.pileup(chrom, pos, pos + 1):
                    # pileup position is 0-based
                    if pileup_column.pos + 1 != pos:
                        continue

                    if ground_truth_variant is not None:
                        true_genotype = ground_truth_variant.samples[sample_id].alleles
                    else:
                        # Assume position without ground-truth genotype has reference allele
                        allele_counts = count_alleles_in_pileup(pileup_column)
                        assumed_ref_allele = max(allele_counts, key=allele_counts.get)
                        true_genotype = (assumed_ref_allele, assumed_ref_allele)

                    efm = pileup_to_efm(pileup_column, true_genotype)
                    out_stream.write(
                        f"pos={pos} true_genotype={true_genotype} "
                        f"pileup={pileup_column.get_query_sequences()}{efm}\n"
                    )
            # gvcf input
            else:
                observed_variant = gvcf_reader.get_variant(chrom, pos)

                # skip positions uncovered by gvcf:
                if observed_variant is None:
                    logger.debug("no information on position for sample")
                    continue

                observed_alleles = get_filtered_alleles_str(observed_variant)
                observed_genotype = observed_variant.samples[0].alleles

                # skip continuous deletion reports (count deletion where it starts)
                if "*" in observed_genotype:
                    continue

                if ground_truth_variant is not None:
                    true_genotype = get_genotype_indices(
                        ground_truth_variant.samples[sample_id]
                    )
                    if true_genotype == "./.":
                        logger.debug("no information on position for ground-truth")
                        continue
                    ground_truth_alleles = get_alleles_str(ground_truth_variant)
                else:
                    # If variant is absent from ground truth file, assume true_genotype is the reference genotype
                    true_genotype = "0/0"
                    first_ref_base = observed_variant.ref[0]
                    ground_truth_alleles = first_ref_base

                allele_counts = count_alleles_in_gvcf(observed_variant)
                conditional_allele_distribution = ConditionalAlleleDistribution(
                    ground_truth_alleles, true_genotype, observed_alleles, allele_counts
                )
                for record in conditional_allele_distribution.get_string_records(
                    chrom, pos
                ):
                    out_stream.write(f"{record}\n")


if __name__ == "__main__":
    main()
