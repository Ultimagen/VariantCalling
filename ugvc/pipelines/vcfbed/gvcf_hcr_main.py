import argparse
import logging
import os
import tempfile
from os.path import dirname

import pybedtools

import ugvc.joint.gvcf_bed as mb


def argparse_gvcf_to_bed() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract bed of high confidence regions from a gvcf file.")
    parser.add_argument("--gvcf", type=str, required=True, help="Path to the gvcf file.")
    parser.add_argument("--bed", type=str, required=True, help="Path to the output bed file.")
    parser.add_argument("--gq_threshold", type=int, default=20, help="GQ threshold.")
    return parser


#
def run(argv: list) -> None:
    """Converts gVCF to high confidence BED file."""
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = argparse_gvcf_to_bed().parse_args(argv[1:])
    logger.info("Step 1: Create intervals of low confidence")
    with tempfile.TemporaryDirectory(dir=dirname(args.bed)) as tmp_dir:
        step1_file = os.path.join(tmp_dir, "high_confidence.bed")
        mb.gvcf_to_bed(args.gvcf, step1_file, args.gq_threshold, gt=True)
        logger.info("Step 2: Merge intervals of high confidence")
        bt = pybedtools.BedTool(step1_file)
        bt = bt.merge().saveas(args.bed)  # type: ignore - issues in pybedtools in vscode
        bt.delete_temporary_history(ask=False)
