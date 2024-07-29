import argparse
import logging
import os
import tempfile

import pybedtools

import ugvc.vcfbed.manipulate_bed as mb


def argparse_gvcf_to_bed() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract bed of high confidence regions from a gvcf file.")
    parser.add_argument("--gvcf", type=str, required=True, help="Path to the gvcf file.")
    parser.add_argument("--bed", type=str, required=True, help="Path to the output bed file.")
    parser.add_argument("--gq_threshold", type=int, default=20, help="GQ threshold.")
    parser.add_argument("--genome_file", type=str, required=True, help="Path to the .sizes file of the genome")
    return parser


def run(argc: list) -> None:
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = argparse_gvcf_to_bed().parse_args(argc)
    logger.info("Step 1: Create intervals of low confidence")
    with tempfile.TemporaryDirectory() as tmp_dir:
        step1_file = os.path.join(tmp_dir, "low_confidence.bed")
        mb.gvcf_to_bed(args.gvcf, step1_file, args.gq_threshold, gt=False)
        logger.info("Step 2: Merge intervals of low confidence")
        step2_file = os.path.join(tmp_dir, "merged_low_confidence.bed")
        pybedtools.BedTool(step1_file).merge().saveas(step2_file)  # type: ignore - issues in pybedtools in vscode
        logger.info("Step 3: Create intervals of high confidence")
        step3_file = args.bed
        pybedtools.BedTool(args.step2_file).complement(genome=args.genome_file).saveas(  # type: ignore
            step3_file
        )  # type: ignore - issues in pybedtools in vscode
