import sys
import pandas as pd
import pysam
from os.path import dirname, basename, join as pjoin
import os
import argparse
import pyfaidx
import logging
from tqdm import tqdm
from tempfile import TemporaryDirectory
import subprocess


logger = logging.getLogger("mrd_utils")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


path = dirname(dirname(dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)


def create_control_signature(
    signature_file,
    reference_fasta,
    control_signature_file_output=None,
    append_python_call_to_header=True,
    delta=1,
    min_distance=5,
    force_overwrite=True,
    progress_bar=False,
):
    """
    Creates a control signature that matches each SNP in the input signature vcf file with an adjacent position with
    the same trinucleotide motif, maintaining the same ref and alt composition. Non-SNP entries are ignored.
    Adds an ORIG_SEQ info in the output vcf indicating the position of the original variant this controls for.

    Parameters
    ----------
    signature_file:
        Input vcf file
    reference_fasta:
        Reference fasta file
        an index (.fai) file is expected to be in the same path
    control_signature_file_output:
        Output path, if None (default) the input file name is used with a ".control.vcf.gz" suffix
    append_python_call_to_header
        Add line to header to indicate this function ran (default True)
    delta
        How many bp to skip when searching for motifs, default 1
    min_distance
        minimum distance in bp from the original position
    force_overwrite
        Force rewrite tbi index of output (if false and output file exists an error will be raised). Default True.
    progress_bar
        Show progress bar (default False)

    Returns
    -------

    """
    ref = pyfaidx.Fasta(reference_fasta)

    assert delta > 0, f"Input parameter delta must be positive, got {delta}"
    if control_signature_file_output is None:
        control_signature_file_output = f"{signature_file}.control.vcf.gz".replace(
            ".vcf.gz.control.", ".control."
        )
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
                header.add_line(
                    f"##python_cmd:create_control_signature=python {' '.join(sys.argv)}"
                )
            with pysam.VariantFile(
                tmp_file, "w", header=header
            ) as f_out:
                for rec in tqdm(
                    f_in, disable=not progress_bar, desc=f"Processing {signature_file}"
                ):
                    is_snv = (
                        len(rec.ref) == 1 and len(rec.alts) == 1 and len(rec.alts[0]) == 1
                    )
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
    pysam.tabix_index(
        control_signature_file_output, preset="vcf", force=force_overwrite
    )


def call_create_control_signature(args_in):
    if args_in.input is None:
        raise ValueError("No input provided")
    create_control_signature(
        signature_file=args_in.input,
        control_signature_file_output=args_in.output,
        reference_fasta=args_in.reference,
        progress_bar=args_in.progress_bar,
    )
    sys.stdout.write("DONE" + os.linesep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_create_control_signature = subparsers.add_parser(
        name="create_control_signature",
        description="""Run full coverage analysis of an aligned bam/cram file""",
    )
    parser_create_control_signature.add_argument(
        "-i", "--input", type=str, required=True, help="input signature vcf file",
    )
    parser_create_control_signature.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output bed file will be written. 
If None (default) the input file name is used with a ".control.vcf.gz" suffix""",
    )
    parser_create_control_signature.add_argument(
        "-r", "--reference", type=str, required=True, help="Reference fasta (local)",
    )
    parser_create_control_signature.add_argument(
        "--progress-bar", action="store_true", help="""Show progress bar""",
    )
    parser_create_control_signature.set_defaults(func=call_create_control_signature)

    args = parser.parse_args()
    args.func(args)
