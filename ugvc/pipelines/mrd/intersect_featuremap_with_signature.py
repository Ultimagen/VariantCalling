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
#    Intersects featuremap vcf-like with pre-defined signature VCF-like
# CHANGELOG in reverse chronological order
import argparse
import os
import sys
from typing import List

import pysam


def __parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="intersect_with_signature", description=run.__doc__
    )
    parser.add_argument(
        "-f", "--featuremap", type=str, required=True, help="""Featuremap vcf file""",
    )
    parser.add_argument(
        "-s", "--signature", type=str, required=True, help="""Signature vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="""Output intersection vcf file (lines from featuremap propagated)""",
    )
    return parser.parse_args(argv[1:])


def run(argv: List[str]):
    """Intersect featuremap and signature vcf files on position and matching ref and alts"""
    args_in = __parse_args(argv)
    intersect_featuremap_with_signature(
        args_in.featuremap, args_in.signature, args_in.output
    )


def intersect_featuremap_with_signature(
    featuremap_file,
    signature_file,
    output_intersection_file,
    append_python_call_to_header=True,
    force_overwrite=True,
    complement=False,
):
    """
    Intersect featuremap and signature vcf files on chrom, position, ref and alts (require same alts), keeping all the
    entries in featuremap. Lines from featuremap propagated to output

    Parameters
    ----------
    featuremap_file
        Of cfDNA
    signature_file
        VCF file, tumor variant calling results
    output_intersection_file
        Output vcf file, .vcf.gz or .vcf extension
    append_python_call_to_header
        Add line to header to indicate this function ran (default True)
    force_overwrite
        Force rewrite tbi index of output (if false and output file exists an error will be raised). Default True.
    complement
        If True, only retain features that do not intersect with signature file - meant for removing germline variants
        from featuremap (default False)

    Returns
    -------

    """
    if (not force_overwrite) and os.path.isfile(output_intersection_file):
        raise OSError(
            f"Output file {output_intersection_file} already exists and force_overwrite flag set to False"
        )
    # build a set of all signature entries, including alts and ref
    signature_entries = set()
    with pysam.VariantFile(signature_file) as f_sig:
        for rec in f_sig:
            signature_entries.add((rec.chrom, rec.pos, rec.ref, rec.alts))
    # Only write entries from featuremap to intersection file if they appear in the signature with the same ref&alts
    try:
        if output_intersection_file.endswith(".gz"):
            output_intersection_file_vcf = output_intersection_file[:-3]
        with pysam.VariantFile(featuremap_file) as f_feat:
            header = f_feat.header
            if append_python_call_to_header is not None:
                header.add_line(
                    f"##python_cmd:intersect_featuremap_with_signature=python {' '.join(sys.argv)}"
                )
            with pysam.VariantFile(
                output_intersection_file_vcf + ".tmp", "w", header=header
            ) as f_int:
                for rec in f_feat:
                    if (
                        (not complement)
                        and (
                            (rec.chrom, rec.pos, rec.ref, rec.alts) in signature_entries
                        )
                    ) or (
                        complement
                        and (
                            (rec.chrom, rec.pos, rec.ref, rec.alts)
                            not in signature_entries
                        )
                    ):
                        f_int.write(rec)
        os.rename(output_intersection_file_vcf + ".tmp", output_intersection_file_vcf)
    finally:
        if "output_intersection_file_vcf" in locals() and os.path.isfile(
            output_intersection_file_vcf + ".tmp"
        ):
            os.remove(output_intersection_file_vcf + ".tmp")
    # index output
    pysam.tabix_index(output_intersection_file_vcf, preset="vcf", force=force_overwrite)
    assert os.path.isfile(output_intersection_file_vcf + ".gz")
