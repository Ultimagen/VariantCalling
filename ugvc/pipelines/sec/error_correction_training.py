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
#    Gather statistics for systematic error correction from gVCFs
# CHANGELOG in reverse chronological order


import argparse
import os.path

from ugvc import logger
from ugvc.sec.allele_counter import count_alleles_in_gvcf
from ugvc.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugvc.vcfbed.buffered_variant_reader import BufferedVariantReader
from ugvc.vcfbed.pysam_utils import get_alleles_str, get_filtered_alleles_str, get_genotype_indices


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
        "--gvcf_file",
        required=True,
        help="path to gvcf file, (for getting the raw aligned reads information)",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if os.path.exists(args.gvcf_file):
        gvcf_reader = BufferedVariantReader(file_name=args.gvcf_file)
    else:
        logger.error("gvcf input does not exist")
        return

    ground_truth_reader = BufferedVariantReader(file_name=args.ground_truth_vcf)
    if args.sample_id not in ground_truth_reader.pysam_reader.header.samples:
        logger.error(
            "sample %s not found in ground truth file %s",
            args.sample_id,
            args.ground_truth_vcf,
        )
        return

    with open(args.output_file, "wt", encoding="utf-8") as out_stream:
        error_correction_training(
            args.relevant_coords,
            ground_truth_reader,
            gvcf_reader,
            args.sample_id,
            out_stream,
        )


def error_correction_training(
    relevant_coords_file: str,
    ground_truth_reader: BufferedVariantReader,
    gvcf_reader: BufferedVariantReader,
    sample_id: str,
    out_stream,
):
    with open(relevant_coords_file, "rt", encoding="utf-8") as relevant_coords:
        for line in relevant_coords:
            fields = line.split("\t")
            chrom, start, end = (
                fields[0],
                int(fields[1]),
                int(fields[2]),
            )  # 0-based coords
            for pos in range(start + 1, end + 1):
                ground_truth_variant = ground_truth_reader.get_variant(chrom, pos)
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
                    true_genotype = get_genotype_indices(ground_truth_variant.samples[sample_id])
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
                for record in conditional_allele_distribution.get_string_records(chrom, pos):
                    out_stream.write(f"{record}\n")


if __name__ == "__main__":
    main()
