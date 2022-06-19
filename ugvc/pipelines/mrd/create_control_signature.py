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
#    Control signature calculation
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from os.path import join as pjoin
from tempfile import TemporaryDirectory

import pyfaidx
import pysam
from tqdm import tqdm


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="create_control_signature", description=run.__doc__)
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="input featuremap files",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input signature vcf file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output bed file will be written.
    If None (default) the input file name is used with a ".control.vcf.gz" suffix""",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        required=True,
        help="Reference fasta (local)",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="""Show progress bar""",
    )
    return parser.parse_args(argv[1:])


def run(argv: list[str]):
    """Creates a vcf file with the same number of variants and identical mutation type distribution as the input vcf,
    in different positions, for MRD background measurement purposes. SNPs only.

    Parameters
    ----------
    argv : List[str]
        Description
    """
    args_in = __parse_args(argv)
    create_control_signature(
        signature_file=args_in.input,
        control_signature_file_output=args_in.output,
        reference_fasta=args_in.reference,
        progress_bar=args_in.progress_bar,
    )
    sys.stdout.write("DONE" + os.linesep)


def create_control_signature(
    signature_file: str,
    reference_fasta: str,
    control_signature_file_output: str = None,
    append_python_call_to_header: bool = True,
    delta: int = 1,
    min_distance: int = 5,
    force_overwrite: bool = True,
    progress_bar: bool = False,
):
    """
    Creates a control signature that matches each SNP in the input signature vcf file with an adjacent position with
    the same trinucleotide motif, maintaining the same ref and alt composition. Non-SNP entries are ignored.
    Adds an ORIG_SEQ info in the output vcf indicating the position of the original variant this controls for.

    Parameters
    ----------
    signature_file: str
        Input vcf file
    reference_fasta: str
        Reference fasta file
        an index (.fai) file is expected to be in the same path
    control_signature_file_output: str
        Output path, if None (default) the input file name is used with a ".control.vcf.gz" suffix
    append_python_call_to_header: bool
        Add line to header to indicate this function ran (default True)
    delta: int
        How many bp to skip when searching for motifs, default 1
    min_distance: int
        minimum distance in bp from the original position
    force_overwrite: bool
        Force rewrite tbi index of output (if false and output file exists an error will be raised). Default True.
    progress_bar: ool
        Show progress bar (default False)


    Raises
    ------
    OSError
        in case the file already exists and function executed with no force overwrite
    """

    ref = pyfaidx.Fasta(reference_fasta)

    assert delta > 0, f"Input parameter delta must be positive, got {delta}"
    if control_signature_file_output is None:
        control_signature_file_output = f"{signature_file}.control.vcf.gz".replace(".vcf.gz.control.", ".control.")
    if (not force_overwrite) and os.path.isfile(control_signature_file_output):
        raise OSError(
            f"Output file {control_signature_file_output} already exists and force_overwrite flag set to False"
        )

    with TemporaryDirectory(prefix=control_signature_file_output) as tmpdir:
        tmp_file = pjoin(tmpdir, control_signature_file_output)
        with pysam.VariantFile(signature_file) as f_in:
            header = f_in.header
            header.info.add("ORIG_POS", 1, "Integer", "Original position of the variant")
            if append_python_call_to_header is not None:
                header.add_line(f"##python_cmd:create_control_signature=python {' '.join(sys.argv)}")
            with pysam.VariantFile(tmp_file, "w", header=header) as f_out:
                for rec in tqdm(f_in, disable=not progress_bar, desc=f"Processing {signature_file}"):
                    is_snv = len(rec.ref) == 1 and len(rec.alts) == 1 and len(rec.alts[0]) == 1
                    if not is_snv:
                        continue

                    chrom = rec.chrom
                    pos = rec.pos
                    motif = ref[chrom][pos - 2 : pos + 1].seq
                    assert (
                        rec.ref == motif[1]
                    ), f"Inconsistency in reference found!\n{rec.chrom} {rec.pos} ref={rec.ref} motif={motif}"

                    new_pos = pos + min_distance  # starting position
                    delta = int(delta)
                    while True:
                        new_motif = ref[chrom][new_pos - 2 : new_pos + 1].seq
                        if (
                            len(new_motif) < 3 or new_motif == "NNN"
                        ):  # reached the end of the chromosome or a reference gap
                            if (
                                delta < 0
                            ):  # if we already looked in the other direction, stop and give up this entry (very rare)
                                break
                            # start looking in the other direction
                            delta = -1 * delta
                            new_pos = pos - min_distance
                            continue
                        if motif == new_motif:  # motif matches
                            rec.pos = new_pos
                            rec.info["ORIG_POS"] = pos
                            f_out.write(rec)
                            break
                        new_pos += delta
        # sort because output can become unsorted and then cannot be indexed
        subprocess.call(f"bcftools sort -Oz -o {control_signature_file_output} {tmp_file}".split())
    # index output
    pysam.tabix_index(control_signature_file_output, preset="vcf", force=force_overwrite)
