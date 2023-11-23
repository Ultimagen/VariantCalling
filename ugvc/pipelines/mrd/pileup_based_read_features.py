from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from enum import Enum

import pysam
from simppl.simple_pipeline import SimplePipeline


class VariantType(Enum):
    SNV = 1
    MONO_BASE_INDEL = 2
    MULTI_BASE_INDEL = 3
    UNIQUE_SNV = 4
    UNIQUE_MONO_BASE_INDEL = 5
    UNIQUE_MULTI_BASE_INDEL = 6
    LOW_AF_SNV = 7
    LOW_AF_MONO_BASE_INDEL = 8
    LOW_AF_MULTI_BASE_INDEL = 9
    COMMON_SNV = 10
    COMMON_MONO_BASE_INDEL = 11
    COMMON_MULTI_BASE_INDEL = 12


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pileup_based_read_features", description="Extract pileup features per read")
    parser.add_argument(
        "--input_bam",
        help="input bam file with aligned reads",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_bed",
        help="bed_file specifying calling regions",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--ref",
        help="reference genome",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input_featuremap",
        help="featuremap input to add new features to",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_pref",
        help="output prefix",
        required=True,
        type=str,
    )

    return parser


class PileupBasedReadFeatures:
    FEATURES_LIST = [
        VariantType.UNIQUE_SNV,
        VariantType.UNIQUE_MONO_BASE_INDEL,
        VariantType.UNIQUE_MULTI_BASE_INDEL,
        VariantType.LOW_AF_SNV,
        VariantType.LOW_AF_MONO_BASE_INDEL,
        VariantType.LOW_AF_MULTI_BASE_INDEL,
        VariantType.COMMON_SNV,
        VariantType.COMMON_MONO_BASE_INDEL,
        VariantType.COMMON_MULTI_BASE_INDEL,
    ]

    FEATURE_NAMES = [f.name for f in FEATURES_LIST]
    MAX_REGION_SIZE = 100000
    PADDING = 300
    AF_THRESHOLD = 0.2

    def __init__(
        self,
        input_featuremap: str,
        input_bam: str,
        input_bed: str,
        reference_fasta: str,
        output_pref: str,
        simple_pipeline_args=(0, 10000, False),
    ):
        self.sp = SimplePipeline(
            simple_pipeline_args[0], simple_pipeline_args[1], debug=simple_pipeline_args[2], print_timing=True
        )

        # Run the shell command to get the access token
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        access_token = result.stdout.strip()  # Extract the access token from the command output

        # Set the environment variable 'GCS_OAUTH_TOKEN' with the obtained access token
        os.environ["GCS_OAUTH_TOKEN"] = access_token

        # if "GCS_OAUTH_TOKEN" in os.environ:
        #     print("Google Cloud credentials environment variable is set.")
        # else:
        #     print("Google Cloud credentials environment variable is not set.")

        self.input_bam = input_bam
        self.input_bed = input_bed
        self.ref = reference_fasta
        self.tmp_region_bed = f"{output_pref}_tmp_region.bed"
        self.tmp_pileup_file = f"{output_pref}.pileup.txt"
        self.featuremap = pysam.VariantFile(input_featuremap)
        self.read_features_tsv = f"{output_pref}.read_features.tsv"
        if os.path.exists(self.read_features_tsv):
            os.remove(self.read_features_tsv)

        self.last_position = 0
        self.last_chr = ""

        for feature_name in self.FEATURE_NAMES:
            self.featuremap.header.add_meta(
                "INFO",
                items=[
                    ("ID", feature_name),
                    ("Number", 1),
                    ("Type", "Integer"),
                    ("Description", f"Count of mismatches of type {feature_name} on a read"),
                ],
            )

        self.output_featuremap = pysam.VariantFile(
            f"{output_pref}.output_featuremap.vcf.gz", "w", header=self.featuremap.header
        )

    def process(self):
        """
        main loop iterating input regions, breaking to chuncks that fit in memory,
        running pileup on each chunk, and counting events of each type from pileups (per read)
        """
        regions = self.__load_regions()

        regions_index = 0
        while regions_index < len(regions):
            region_size = 0
            tmp_regions = []
            while region_size < self.MAX_REGION_SIZE:
                if regions_index >= len(regions):
                    break
                chrom, start, end = regions[regions_index]
                tmp_regions.append((chrom, start, end))
                region_size += end - start + (self.PADDING * 2)
                regions_index += 1

            # Now we have tmp_regions with region_size >= MAX_REGION_SIZE

            # if last region is very large, take it out of list and process later
            if region_size > self.MAX_REGION_SIZE * 1.5:
                last_region_chrom, last_region_start, last_region_end = tmp_regions.pop()

            # if has other small regions
            if len(tmp_regions) > 0:
                # write small regions to bed file
                with open(self.tmp_region_bed, "w", encoding="utf-8") as tmp_bed:
                    for chrom, start, end in tmp_regions:
                        tmp_bed.write(f"{chrom}\t{start - self.PADDING}\t{end + self.PADDING}\n")
                self.sp.print_and_run(
                    f'samtools mpileup {self.input_bam} -Q 0 --ff "" -l {self.tmp_region_bed} '
                    f"-f {self.ref} --output-extra QNAME -o {self.tmp_pileup_file}"
                )
                read_features, events_per_read = self.__calc_read_features_from_pileup()
                # sys.stderr.write(f'{read_alts}\n')

                for chrom, start, end in tmp_regions:
                    print("working on region: chrom,start,end:", chrom, start, end)
                    self.__add_features_to_featuremap(chrom, start, end, read_features, events_per_read)

            # if last region was very large, break it into smaller regions
            if region_size > self.MAX_REGION_SIZE * 1.5:
                for start_ in range(last_region_start, last_region_end, self.MAX_REGION_SIZE):
                    end_ = min(start_ + self.MAX_REGION_SIZE, last_region_end)
                    padded_start = start_ - self.PADDING
                    padded_end = end_ + self.PADDING
                    self.sp.print_and_run(
                        f'samtools mpileup {self.input_bam} -Q 0 --ff "" '
                        f"-r {last_region_chrom}:{padded_start}-{padded_end} "
                        f"-f {self.ref} --output-extra QNAME -o {self.tmp_pileup_file}"
                    )
                    read_features, events_per_read = self.__calc_read_features_from_pileup()
                    self.__add_features_to_featuremap(chrom, start_, end_, read_features, events_per_read)

    @staticmethod
    def parse_pileup(
        pileup_alleles: str, read_names: list
    ) -> tuple[dict[str, tuple[str, VariantType]], dict[str, int]]:
        """
        Parse the pileup_string of a specific position, associating an allele per read
        Return:
        read_alleles:  read_name -> (allele, variant_type)
        allele_counts: allele -> count
        """
        pileup_alleles = pileup_alleles.replace("$", "").upper()
        allele_counts = defaultdict(int)
        allele_counts["R"] = pileup_alleles.count(".") + pileup_alleles.count(",")
        char_index = 0
        read_index = 0
        read_alleles = {}
        while char_index < len(pileup_alleles):
            c = pileup_alleles[char_index]
            if c in {".", ",", "*"}:
                read_index += 1
            elif c == "^":
                char_index += 1
            elif c in {"+", "-"}:
                char_index += 1
                indel_length = int(re.match(r"\d+", pileup_alleles[char_index:])[0])
                indel_allele = c + pileup_alleles[char_index + 1 : char_index + 1 + indel_length]
                vtype = VariantType.MONO_BASE_INDEL if len(set(indel_allele)) == 2 else VariantType.MULTI_BASE_INDEL
                char_index += indel_length + len(str(indel_length)) - 1
                # Read is last read since indel refers to the same read as the base before it
                read_name = read_names[read_index - 1]
                if read_name not in read_alleles:  # deduplicate realigned bam
                    read_alleles[read_name] = (indel_allele, vtype)
                    allele_counts[indel_allele] += 1
            else:
                read_name = read_names[read_index]
                read_alleles[read_name] = (c, VariantType.SNV)
                allele_counts[c] += 1
                read_index += 1
            char_index += 1

        return read_alleles, allele_counts

    def __load_regions(self) -> list[tuple[str, int, int]]:
        """
        eliminate overlaps and load non-overlapping regions to memory as list of tuples
        """
        no_overlap_bed = self.tmp_region_bed

        self.sp.print_and_run(f"bedtools merge -i {self.input_bed} > {no_overlap_bed}")
        regions = []
        with open(no_overlap_bed, encoding="utf-8") as bed:
            for line in bed:
                chrom, start, end = line.split()
                regions.append((chrom, int(start), int(end)))
        return regions

    def __calc_read_features_from_pileup(self) -> tuple[dict, dict]:
        """
        iterate chunk's pileup and count read events according to event categories (VariantType)
        Return:
        read_features: dict[read_name, dict[variant_type_int, count]]
                        counts of various VariantTypes (as integer) occuring per read
        events_per_read: dict[read_name, list[tuple[chrom, pos, allele, allele_count, allele_frequency]]]
        """
        events_per_read = defaultdict(list)
        read_features = defaultdict(dict)
        with open(self.tmp_pileup_file, encoding="utf-8") as pu:
            for line in pu:
                chrom, pos, _, _, alleles_str, _, read_names = line.split("\t")

                read_names = read_names.strip().split(",")

                read_alleles, allele_counts = self.parse_pileup(alleles_str, read_names)
                for read_name, allele_and_type in read_alleles.items():
                    allele = allele_and_type[0]
                    vtype_int = allele_and_type[1].value
                    allele_count = allele_counts[allele]
                    allele_frequency = allele_count / len(read_names)
                    if read_name not in events_per_read:
                        events_per_read[read_name] = []
                    events_per_read[read_name].append((chrom, int(pos), allele, allele_count, allele_frequency))
                    # classify as unique low_af or common
                    # unique
                    if allele_count == 1:
                        vtype_int += 3
                    # low_af
                    elif allele_frequency < self.AF_THRESHOLD:
                        vtype_int += 6
                    # common
                    else:
                        vtype_int += 9
                    if vtype_int in read_features[read_name]:
                        read_features[read_name][vtype_int] += 1
                    else:
                        read_features[read_name][vtype_int] = 1

        with open(self.read_features_tsv, "a", encoding="utf-8") as of:
            for read, info in read_features.items():
                vtype_counts = []
                for vtype_int in self.FEATURES_LIST:
                    if vtype_int.value in info:
                        vtype_counts.append(info[vtype_int.value])
                    else:
                        vtype_counts.append(0)
                record = [read] + [str(x) for x in vtype_counts]
                of.write("\t".join(record))
                of.write("\n")
                read_features[read] = vtype_counts
        # for read, info in events_per_read.items():
        #     print(read, info)
        return read_features, events_per_read

    def __add_features_to_featuremap(
        self, chrom: str, start: int, end: int, read_features: dict[str, list[int]], events_per_read: dict[str, tuple]
    ):
        """
        subtract the featuremap event from the read_features count of this read
        (using events_per_read to re-catagorize it)
        write the finalized feature counts as info feilds in the output vcf featuremap
        """
        # Run the shell command to get the access token
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        access_token = result.stdout.strip()  # Extract the access token from the command output

        # Set the environment variable 'GCS_OAUTH_TOKEN' with the obtained access token
        os.environ["GCS_OAUTH_TOKEN"] = access_token

        # if "GCS_OAUTH_TOKEN" in os.environ:
        #     print("Google Cloud credentials environment variable is set.")
        # else:
        #     print("Google Cloud credentials environment variable is not set.")

        for record in self.featuremap.fetch(chrom, start, end):
            if record.chrom == self.last_chr and record.pos < self.last_position:
                continue
            if record.chrom != self.last_chr:
                self.last_chr = record.chrom
            self.last_position = record.pos

            read_name = record.info["X_RN"]
            if read_name in read_features:
                features = read_features[read_name]
                read_events = events_per_read[read_name]
                relevant_event = [e for e in read_events if e[1] == record.pos]
                if len(relevant_event) == 0:
                    sys.stderr.write(f"missing featuremap event: {record}")
                    continue
                relevant_event = relevant_event[0]
                (_, _, _, allele_count, allele_frequency) = relevant_event
                if allele_count == 1:
                    features[VariantType.UNIQUE_SNV.value - 4] -= 1
                elif allele_frequency < self.AF_THRESHOLD:
                    features[VariantType.LOW_AF_SNV.value - 4] -= 1
                else:
                    features[VariantType.COMMON_SNV.value - 4] -= 1
                for i, feature_value in enumerate(features):
                    if feature_value < 0:
                        raise RuntimeError("Found negative feature count: {features} for read {read_name}")
                    feature_name = self.FEATURE_NAMES[i]
                    record.info[feature_name] = feature_value
                self.output_featuremap.write(record)
            else:
                sys.stderr.write(f"missing read info {read_name} at {record.chrom} {record.pos}\n")
                for feature_name in self.FEATURE_NAMES:
                    record.info[feature_name] = 0


def run(argv):
    """
    Add pileup-based features to featuremap (#unique/common snvs and indels)
    """
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])
    PileupBasedReadFeatures(
        input_bam=args.input_bam,
        input_featuremap=args.input_featuremap,
        input_bed=args.input_bed,
        reference_fasta=args.ref,
        output_pref=args.output_pref,
    ).process()


if __name__ == "__main__":
    run(sys.argv)
