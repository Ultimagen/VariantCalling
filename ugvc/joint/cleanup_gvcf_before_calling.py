from __future__ import annotations

import tqdm
from pysam import VariantFile

import ugvc.vcfbed.pysam_utils as pu


def filterOverlappingNoneGVCFs_v2(input_gvcf: str, output_gvcf: str) -> tuple(int, int):
    """Filters non-called variants that overlap other variants from DeepVariant gVCFs.
    In the context of GLNexus joint calling, non-called deletion variants (./.) that
    overlap called variants can generate missed variant calls in joint calling.
    For now we introduced a hack that removed missed-genotype variants
    that overlap called variants.

    Parameters
    ----------
    input_gvcf : str
        input gvcf
    output_gvcf : str
        cleaned up gvcf

    Returns
    -------
    tuple(int,int)
        Number of written, number of removed records

    """
    with VariantFile(input_gvcf) as vcf_in, VariantFile(output_gvcf, mode="w", header=vcf_in.header) as vcf_out:

        # once we read a variant that can overlap later variants, we keep a buffer
        # of overlapping variants
        buffer: list = []
        buffer_chrom = ""
        buffer_span = 0

        # special logic is applied to the buffer only if it contains a homozyous alt variant
        called_var_in_buffer = False
        count_skipped = 0
        count_written = 0
        for rec in tqdm.tqdm(vcf_in):
            # if we finished the block of overlapping variants - we dump the buffer
            if rec.chrom != buffer_chrom or rec.pos > buffer_span:  # if we are out of the buffer
                while buffer:
                    tmp = buffer.pop(0)
                    if not called_var_in_buffer or tmp.samples[0]["GT"] != (None, None):
                        vcf_out.write(tmp)
                        count_written += 1
                    # skip all variants that overlap hom alt and have GT not called
                    else:  # (tmp.samples[0]['GT']==(None,None)):
                        count_skipped += 1
                        continue
                    buffer_chrom = ""
                    buffer_span = 0
                called_var_in_buffer = False
            # now for the new deletion
            dels_in_rec = pu.is_deletion(rec)
            lens_in_rec = pu.indel_length(rec)
            del_lens = [lens_in_rec[i] for i, x in enumerate(dels_in_rec) if x] + [0]
            if buffer:  # either it is covered by the buffer
                buffer.append(rec)
                if max(del_lens) > 0:
                    buffer_span = max(buffer_span, rec.pos + max(del_lens))
            elif max(del_lens) > 0:  # or if it is a deletion - we will append it to the buffer
                buffer.append(rec)
                buffer_chrom = rec.chrom
                buffer_span = rec.pos + max(del_lens)
            else:
                vcf_out.write(rec)
                count_written += 1
            if buffer and (rec.samples[0]["GT"][0] is not None) and (rec.samples[0]["GT"] != (0, 0)):
                called_var_in_buffer = True

        # dump buffer at the end
        while buffer:
            tmp = buffer.pop(0)
            if not called_var_in_buffer or tmp.samples[0]["GT"] != (None, None):
                vcf_out.write(tmp)
                count_written += 1
            # skip all variants that overlap hom alt and have GT not called
            else:  # (tmp.samples[0]['GT']==(None,None)):
                count_skipped += 1
                continue
    return count_written, count_skipped
