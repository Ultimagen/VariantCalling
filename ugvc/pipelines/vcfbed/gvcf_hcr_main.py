import argparse
import logging
import os
import tempfile
from os.path import dirname

import pybedtools

import ugvc.joint.gvcf_bed as mb
from ugvc.vcfbed.interval_file import IntervalFile


def argparse_gvcf_to_bed() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract bed of high confidence regions from a gvcf file.")
    parser.add_argument("--gvcf", type=str, required=True, help="Path to the gvcf file.")
    parser.add_argument("--bed", type=str, required=True, help="Path to the output bed file.")
    parser.add_argument("--gq_threshold", type=int, default=20, help="GQ threshold.")
    parser.add_argument("--genome_file", type=str, required=True, help="Path to the .sizes file of the genome")
    parser.add_argument("--calling_region", type=str, required=False, help="Calling regions if not whole genome")
    return parser


def run(argc: list) -> None:
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = argparse_gvcf_to_bed().parse_args(argc)
    logger.info("Step 1: Create intervals of low confidence")
    with tempfile.TemporaryDirectory(dir=dirname(args.bed)) as tmp_dir:
        step1_file = os.path.join(tmp_dir, "low_confidence.bed")
        mb.gvcf_to_bed(args.gvcf, step1_file, args.gq_threshold, gt=False)
        logger.info("Step 2: Merge intervals of low confidence")
        step2_file = os.path.join(tmp_dir, "merged_low_confidence.bed")
        bt = pybedtools.BedTool(step1_file)
        bt = bt.merge().saveas(step2_file)  # type: ignore - issues in pybedtools in vscode
        bt.delete_temporary_history(ask=False)
        logger.info("Step 3: Create intervals of high confidence")

        if not args.calling_region:
            step3_file = args.bed
        else:
            step3_file = os.path.join(tmp_dir, "high_confidence.bed")
        bt = pybedtools.BedTool(step2_file)
        bt = bt.complement(g=args.genome_file).saveas(  # type: ignore
            step3_file
        )  # type: ignore - issues in pybedtools in vscode
        bt.delete_temporary_history(ask=False)
        if args.calling_region:
            intv = IntervalFile(interval=args.calling_region)
            step4_file = args.bed
            logger.info("Step 4: Intersect with calling region")
            bt = pybedtools.BedTool(step3_file)
            bt = bt.intersect(b=intv.as_bed_file()).saveas(step4_file)
            bt.delete_temporary_history(ask=False)
