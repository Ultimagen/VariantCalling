from __future__ import annotations

import pysam
import tqdm.auto as tqdm

from ugbio_core.bed_writer import BedWriter


def gvcf_to_bed(gvcf_file: str, bed_file: str, gq_threshold: int = 20, gt: bool = True) -> int:
    """Select records with GQ >= (or <) gq_threshold and write them to a bed file.
    If gt is True, select records with GQ >= gq_threshold. If gt is False, select records with GQ < gq_threshold.
    The output is written to a bed file.
    Parameters
    ----------
    gvcf_file : str
        Path to the gvcf file.
    bed_file : str
        Path to the bed file.
    gq_threshold : int
        GQ threshold.
    gt : bool
        If True, select records with GQ >= gq_threshold. If False, select records with GQ < gq_threshold.

    Returns
    -------
    int
        Number of skipped records
    """
    with pysam.VariantFile(gvcf_file) as vcf:
        bed = BedWriter(bed_file)
        extent = -1
        last_chrom = ""
        skipped = 0
        chrom = last_chrom
        end = extent
        start = extent

        for record in tqdm.tqdm(vcf.fetch()):
            # If the record is a reference block, write the entire block to the bed file
            # however, if the record is a "refCall" deletion, it will only write the first base
            # The other bases will be marked as "uncertain" because they will be taken from the
            # reference blocks of the gVCF that the efficient deepVariant outputs
            if len(str(record.ref)) > 1 and (
                ("GQ" not in record.samples[0])
                or record.samples[0]["GT"] == (0, 0)
                or record.samples[0]["GT"] == (None, None)
            ):
                chrom = record.chrom
                start = record.start  # VCF is 0-based
                end = record.start + 1  # gVCF end position
            else:
                chrom = record.chrom
                start = record.start  # VCF is 0-based
                end = record.stop  # gVCF end position

            if chrom == last_chrom and start < extent:
                skipped += 1
                continue
            if chrom != last_chrom or extent < end:
                last_chrom = chrom
                extent = end

            if gt:
                if "GQ" in record.samples[0] and record.samples[0]["GQ"] >= gq_threshold:
                    bed.write(chrom, start, end)
            elif not gt:
                if "GQ" not in record.samples[0] or record.samples[0]["GQ"] < gq_threshold:
                    bed.write(chrom, start, end)
    return skipped
