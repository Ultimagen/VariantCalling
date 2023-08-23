from __future__ import annotations

import itertools
import os
import tempfile
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pyBigWig as pbw
import pyfaidx
import pysam
from joblib import Parallel, delayed

import ugvc.dna.utils as dnautils
import ugvc.flow_format.flow_based_read as flowBasedRead
import ugvc.utils.misc_utils as utils
from ugvc.dna.format import (
    CYCLE_SKIP,
    CYCLE_SKIP_DTYPE,
    CYCLE_SKIP_STATUS,
    IS_CYCLE_SKIP,
    NON_CYCLE_SKIP,
    POSSIBLE_CYCLE_SKIP,
    UNDETERMINED_CYCLE_SKIP,
)
from ugvc.vcfbed import bed_writer

UNDETERMINED = "NA"


class VcfAnnotator(ABC):
    """
    An abstract base class for annotating VCF files.
    """

    @abstractmethod
    def __init__(self):
        """
        Initializer of the base class.
        Derived classes should override this method if they require additional parameters.
        """
        pass

    @abstractmethod
    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Method to edit VCF header, accepts a pysam header, modifies and returns it.
        Derived classes should override this method to provide specific functionality.

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF file header to be edited.

        Returns
        -------
        pysam.VariantHeader
            Edited VCF file header.
        """
        pass

    @abstractmethod
    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """
        Method to process VCF records. Accepts a list of pysam VariantRecords, modifies each and returns the list
        Derived classes should override this method to provide specific functionality.

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            List of VCF records to be processed.

        Returns
        -------
        list[pysam.VariantRecord]
            Processed VCF records.
        """
        pass

    @staticmethod
    def merge_temp_files(contig_output_vcfs: list[str], output_path: str, header: pysam.VariantHeader):
        """
        Static method to merge temporary output files and write to the final output file.

        Args:
        contigs : list[str]
            List of contig names.
        output_path : str
            Path to the final output file.
        header : pysam.VariantHeader
            VCF file header.
        """
        try:
            # Open the final output file
            with pysam.VariantFile(output_path, "w", header=header) as vcf_out:
                # Loop over each contig
                for temp_file_path in contig_output_vcfs:
                    # Open the temp file
                    with pysam.VariantFile(temp_file_path, "r") as vcf_contig:
                        # Write each record from the temp file to the final output file
                        for record in vcf_contig:
                            vcf_out.write(record)
        finally:
            # Loop over each contig again
            for temp_file_path in contig_output_vcfs:
                # If the temp file exists, remove it
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    @staticmethod
    def process_vcf(
        annotators: list[VcfAnnotator],
        input_path: str,
        output_path: str,
        chunk_size: int = 10000,
        multiprocess_contigs: bool = False,
    ):
        """
        Static method to process a VCF file in chunks with multiple VcfAnnotator objects.
        Optionally runs in parallel over different contigs.

        Parameters
        ----------
        annotators : list[VcfAnnotator]
            List of VcfAnnotator objects.
        input_path : str
            Path to the input VCF file.
        output_path : str or pysam.VariantFile
            Path to the output file. If provided, a separate output file is created for each contig.
        chunk_size : int, optional
            The chunk size. Defaults to 10000.
        multiprocess_contigs : bool, optional
            If True, runs in parallel over different contigs. Defaults to False.


        """
        # Open the input VCF file
        with pysam.VariantFile(input_path) as vcf_in:
            # Edit the header
            new_header = vcf_in.header
            for annotator in annotators:
                new_header = annotator.edit_vcf_header(new_header)

            # Get the contigs
            contigs = list(vcf_in.header.contigs)

            if multiprocess_contigs:
                with tempfile.TemporaryDirectory(dir=os.path.dirname(output_path)) as temp_dir:
                    # determine output paths
                    tmp_output_paths = {c: os.path.join(temp_dir, c) for c in contigs}
                    # Process the contigs in parallel
                    Parallel(n_jobs=-1)(
                        delayed(VcfAnnotator._process_contig)(
                            vcf_in, tmp_output_paths[contig], annotators, contig, chunk_size, header=new_header
                        )
                        for contig in contigs
                    )

                    # Merge the temporary output files into the final output file and remove them
                    VcfAnnotator.merge_temp_files(tmp_output_paths.values(), output_path, new_header)
            else:
                # Process the contigs one at a time
                with pysam.VariantFile(output_path, "w", header=new_header) as vcf_out:
                    for contig in contigs:
                        VcfAnnotator._process_contig(vcf_in, vcf_out, annotators, contig, chunk_size)
            # Create a tabix index for the final output file
            pysam.tabix_index(output_path, preset="vcf", force=True)

    @staticmethod
    def _process_contig(
        vcf_in: pysam.VariantFile,
        vcf_out: [pysam.VariantFile | str],
        annotators: list[VcfAnnotator],
        contig: str,
        chunk_size: int,
        header: pysam.VariantHeader = None,
    ):
        """
        Static helper method to process a single contig in chunks.

        Args:
        vcf_in : pysam.VariantFile | str
            The input VCF file, path or pysam object.
        annotators : list[VcfAnnotator]
            List of VcfAnnotator objects.
        contig : str
            The contig to process.
        chunk_size : int
            The chunk size.
        output_path : str, optional
            Path to the output file. If provided, a separate output file is created for each contig.
        """
        close_when_finished = False
        try:
            if isinstance(vcf_out, str):
                vcf_out = pysam.VariantFile(vcf_out, "w", header=header)
                close_when_finished = True
            records = []
            for record in vcf_in.fetch(contig):
                records.append(record)

                if len(records) == chunk_size:
                    for annotator in annotators:
                        records = annotator.process_records(records)

                    for record in records:
                        vcf_out.write(record)

                    records = []

            # Process the remaining records
            if records:
                for annotator in annotators:
                    records = annotator.process_records(records)

                for record in records:
                    vcf_out.write(record)
        finally:
            if close_when_finished and isinstance(vcf_out, pysam.VariantFile):
                vcf_out.close()


# pylint: disable=too-many-instance-attributes
class RefContextVcfAnnotator(VcfAnnotator):
    def __init__(
        self,
        ref_fasta: str,
        flow_order=str,
        motif_length_to_annotate: int = 3,
        max_hmer_length: int = 20,
    ):
        """
        Annotator to add reference context to VCF records, only modifies biallelic SNVs.
        The following are added to the INFO field:
        - trinuc_context: reference trinucleotide context
        - prev_Nbp: N bases in the reference before the variant, N=length motif_length_to_annotate
        - next_Nnp: N bases in the reference after the variant, N=length motif_length_to_annotate
        - hmer_context_ref: reference homopolymer context, up to length max_hmer_length
        - hmer_context_alt: homopolymer context in the ref allele (assuming the variant considered only),
            up to length max_hmer_length
        - is_cycle_skip: True if the SNV is a cycle skip

        Parameters
        ----------
        ref_fasta : str
            Path to the reference FASTA file.
        flow_order : str
            Flow order of the flow cell.
        motif_length_to_annotate : int, optional
            The length of the motif to annotate context up to (prev_Nbp / next_Nbp). Defaults to 4.
        max_hmer_length : int, optional
            The maximum length of the homopolymer to annotate context up to. Defaults to 20.
        """
        # check inputs
        assert len(flow_order) == 4, f"Flow order must be of length 4, got {flow_order}"
        assert os.path.isfile(ref_fasta), f"Reference FASTA file not found: {ref_fasta}"

        # save inputs
        self.ref_fasta = ref_fasta
        self.motif_length_to_annotate = motif_length_to_annotate
        self.max_hmer_length = max_hmer_length
        self.flow_order = flow_order

        # init accesory objects
        self.cycle_skip_dataframe = get_cycle_skip_dataframe(flow_order)
        self.faidx_ref = pyfaidx.Fasta(self.ref_fasta, build_index=False, rebuild=False)

        # info field names
        self.TRINUC_CONTEXT = "trinuc_context"
        self.HMER_CONTEXT_REF = "hmer_context_ref"
        self.HMER_CONTEXT_ALT = "hmer_context_alt"
        self.CYCLE_SKIP_FLAG = "is_cycle_skip"
        self.PREV_N_BP = f"prev_{self.motif_length_to_annotate}bp"
        self.NEXT_N_BP = f"next_{self.motif_length_to_annotate}bp"
        self.info_fields_to_add = [
            self.TRINUC_CONTEXT,
            self.HMER_CONTEXT_REF,
            self.HMER_CONTEXT_ALT,
            self.PREV_N_BP,
            self.NEXT_N_BP,
        ]  # self.CYCLE_SKIP_FLAG not included because it's a flag, only there if True

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to include new fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            Modified VCF header

        """
        header.add_line(
            "##ugvc_ugvc/vcfbed/variant_annotation.py_RefContextVcfAnnotator="
            f"ref:{os.path.basename(self.ref_fasta)}"
            f"_motif_length_to_annotate:{self.motif_length_to_annotate}"
            f"_max_hmer_length:{self.max_hmer_length}"
        )
        header.add_line(
            f"##INFO=<ID={self.TRINUC_CONTEXT}," 'Number=1,Type=String,Description="reference trinucleotide context">'
        )
        header.add_line(
            f"##INFO=<ID={self.PREV_N_BP},"
            f'Number=1,Type=String,Description="{self.motif_length_to_annotate}'
            'bases in the reference before variant">'
        )
        header.add_line(
            f"##INFO=<ID={self.NEXT_N_BP},"
            f'Number=1,Type=String,Description="{self.motif_length_to_annotate}'
            'bases in the reference after variant">'
        )
        header.add_line(
            f"##INFO=<ID={self.HMER_CONTEXT_REF},"
            f'Number=1,Type=Integer,Description="reference homopolymer context, '
            f'up to length {self.max_hmer_length}">'
        )
        header.add_line(
            f"##INFO=<ID={self.HMER_CONTEXT_ALT},"
            f'Number=1,Type=Integer,Description="homopolymer context in the ref allele '
            f'(assuming the variant considered only), up to length {self.max_hmer_length}">'
        )
        header.add_meta("INFO", 
                        items=[("ID", self.CYCLE_SKIP_FLAG), 
                        ("Number", 0),
                        ("Type", "Flag"), 
                        ("Description", "is the SNV a cycle skip")])

        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """


        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of updated VCF records
        """
        records_out = [None] * len(records)
        for j, record in enumerate(records):
            if is_biallelic_snv(record):
                # get motif
                ref_around_snv = get_motif_around_snv(record, self.max_hmer_length, self.faidx_ref)
                central_base_ind = len(ref_around_snv) // 2
                trinuc_ref = ref_around_snv[central_base_ind - 1 : central_base_ind + 2]
                # create sequence of alt allele
                alt_around_snv_list = list(ref_around_snv)
                alt_around_snv_list[central_base_ind] = record.alts[0]
                alt_around_snv = "".join(alt_around_snv_list)
                trinuc_alt = alt_around_snv[central_base_ind - 1 : central_base_ind + 2]
                # assign to record
                record.info[self.HMER_CONTEXT_REF] = get_hmer_of_central_base(ref_around_snv)
                record.info[self.HMER_CONTEXT_ALT] = get_hmer_of_central_base(alt_around_snv)
                record.info[self.TRINUC_CONTEXT] = trinuc_ref
                record.info[self.PREV_N_BP] = ref_around_snv[
                    central_base_ind - self.motif_length_to_annotate : central_base_ind
                ]
                record.info[self.NEXT_N_BP] = ref_around_snv[
                    central_base_ind + 1 : central_base_ind + self.motif_length_to_annotate + 1
                ]
                is_cycle_skip = self.cycle_skip_dataframe.loc[(trinuc_ref, trinuc_alt), IS_CYCLE_SKIP]
                record.info[self.CYCLE_SKIP_FLAG] = is_cycle_skip

                # make sure all the info fields are present
                for info_field in self.info_fields_to_add:
                    assert (
                        info_field in record.info
                    ), f"INFO field {info_field} was supposed to be added to VCF record but was not found"
            records_out[j] = record

        return records_out


def classify_indel(concordance: pd.DataFrame) -> pd.DataFrame:
    """Classifies indel as insertion or deletion

    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe of concordances

    Returns
    -------
    pd.DataFrame:
        Modifies dataframe by adding columns "indel_classify" and "indel_length"
    """

    def classify(x):
        if not x["indel"]:
            return None
        if len(x["ref"]) < max(len(y) for y in x["alleles"]):
            return "ins"
        return "del"

    concordance["indel_classify"] = concordance.apply(classify, axis=1, result_type="reduce")
    concordance["indel_length"] = concordance.apply(
        lambda x: max(abs(len(y) - len(x["ref"])) for y in x["alleles"]),
        axis=1,
        result_type="reduce",
    )
    return concordance


def is_hmer_indel(concordance: pd.DataFrame, fasta_file: str) -> pd.DataFrame:
    """Checks if the indel is hmer indel and outputs its length
    Note: The length of the shorter allele is output.

    Parameters
    ----------
    concordance: pd.DataFrame
        Dataframe of concordance or of VariantCalls. Should contain a collumn "indel_classify"
    fasta_file: str
        FAI indexed fasta file

    Returns
    -------
    pd.DataFrame
        Adds column hmer_indel_length, hmer_indel_nuc
    """

    fasta_idx = pyfaidx.Fasta(fasta_file, build_index=False, rebuild=False)

    def _is_hmer(rec, fasta_idx):
        if not rec["indel"]:
            return (0, None)

        if rec["indel_classify"] == "ins":
            alt = [x for x in rec["alleles"] if x != rec["ref"]][0][1:]
            if len(set(alt)) != 1:
                return (0, None)
            if fasta_idx[rec["chrom"]][rec["pos"]].seq.upper() != alt[0]:
                return (0, None)
            return (
                dnautils.hmer_length(fasta_idx[rec["chrom"]], rec["pos"]),
                alt[0],
            )

        if rec["indel_classify"] == "del":
            del_seq = rec["ref"][1:]
            if len(set(del_seq)) != 1:
                return (0, None)
            if fasta_idx[rec["chrom"]][rec["pos"] + len(rec["ref"]) - 1].seq.upper() != del_seq[0]:
                return (0, None)

            return (
                len(del_seq) + dnautils.hmer_length(fasta_idx[rec["chrom"]], rec["pos"] + len(rec["ref"]) - 1),
                del_seq[0],
            )
        return (0, None)

    results = concordance.apply(lambda x: _is_hmer(x, fasta_idx), axis=1, result_type="reduce")
    concordance["hmer_indel_length"] = [x[0] for x in results]
    concordance["hmer_indel_nuc"] = [x[1] for x in results]
    return concordance


def get_hmer_of_central_base(sequence: str) -> int:
    """
    Get the length of the homopolymer spanning the central base in the given sequence.
    Examples:
    ACA -> 1
    ACC -> 2
    ACCCT -> 3
    AAAAAT -> 5

    Parameters
    ----------
    sequence : str
        The sequence to check. Must be of odd length, so the central base is well defined.

    Returns
    -------
    int
        The length of the homopolymer run of the central base.
    """
    assert isinstance(sequence, str)
    assert len(sequence) % 2 == 1
    assert len(sequence) >= 1
    hmer_lengths = [sum(1 for _ in x[1]) for x in itertools.groupby(sequence)]
    central_hmer = hmer_lengths[np.argmax(np.cumsum(hmer_lengths) > len(sequence) // 2)]
    return central_hmer


def is_biallelic_snv(record: pysam.VariantRecord) -> bool:
    """
    Check if a given VCF record is a biallelic SNV.

    Parameters
    ----------
    record : pysam.VariantRecord
        The VCF record to check.

    Returns
    -------
    bool
        True if the record is a biallelic SNV, False otherwise.
    """
    assert isinstance(record, pysam.VariantRecord)
    return len(record.ref) == 1 and len(record.alts) == 1 and len(record.alts[0]) == 1


def get_motif_around_snv(record: pysam.VariantRecord, size: int, faidx: pyfaidx.Fasta):
    """ "
    Extract sequence around an SNV, of size "size" on each side with the central base being the ref base

    Parameters
    ----------
    record: pysam.VariantRecord
        Record, must be an SNV or result will be wrong
    size: int
        Size of motif on each side
    faidx: pyfaidx.Fasta
        Fasta index

    Returns
    -------
    """
    assert isinstance(record, pysam.VariantRecord), f"record must be pysam.VariantRecord, got {type(record)}"
    size = int(size)
    assert size > 0, f"size must be positive, got {size}"

    chrom = faidx[record.chrom]
    pos = record.pos
    return chrom[pos - size - 1 : pos + size].seq.upper()


def get_motif_around(concordance: pd.DataFrame, motif_size: int, fasta: str) -> pd.DataFrame:
    """Extract sequence around the indel

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    motif_size: int
        Size of motif on each side
    fasta: str
        Indexed fasta

    Returns
    -------
    pd.DataFrame
        DataFrame. Adds "left_motif" and right_motif
    """

    def _get_motif_around_snp(rec, size, faidx):
        chrom = faidx[rec["chrom"]]
        pos = rec["pos"]
        return (
            chrom[pos - size - 1 : pos - 1].seq.upper(),
            chrom[pos : pos + size].seq.upper(),
        )

    def _get_motif_around_non_hmer_indel(rec, size, faidx):
        chrom = faidx[rec["chrom"]]
        pos = rec["pos"]
        return (
            chrom[pos - size : pos].seq.upper(),
            chrom[pos + len(rec["ref"]) - 1 : pos + len(rec["ref"]) - 1 + size].seq.upper(),
        )

    def _get_motif_around_hmer_indel(rec, size, faidx):
        chrom = faidx[rec["chrom"]]
        pos = rec["pos"]
        hmer_length = rec["hmer_indel_length"]
        return (
            chrom[pos - size : pos].seq.upper(),
            chrom[pos + hmer_length : pos + hmer_length + size].seq.upper(),
        )

    def _get_motif(rec, size, faidx):
        if rec["indel"] and rec["hmer_indel_length"] > 0:
            return _get_motif_around_hmer_indel(rec, size, faidx)
        if rec["indel"] and rec["hmer_indel_length"] == 0:
            return _get_motif_around_non_hmer_indel(rec, size, faidx)
        return _get_motif_around_snp(rec, size, faidx)

    faidx = pyfaidx.Fasta(fasta, build_index=False, rebuild=False)
    tmp = concordance.apply(lambda x: _get_motif(x, motif_size, faidx), axis=1, result_type="reduce")
    concordance["left_motif"] = [x[0] for x in list(tmp)]
    concordance["right_motif"] = [x[1] for x in list(tmp)]
    return concordance


def get_gc_content(concordance: pd.DataFrame, window_size: int, fasta: str) -> pd.DataFrame:
    """Extract sequence around the indel

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    window_size: int
        Size of window for GC calculation (around start pos of variant)
    fasta: str
        Indexed fasta

    Returns
    -------
    pd.DataFrame
        DataFrame. Adds "left_motif" and right_motif
    """

    def _get_gc(rec, size, faidx):
        chrom = faidx[rec["chrom"]]
        beg = rec["pos"] - int(size / 2)
        end = beg + size
        seq = chrom[beg:end].seq.upper()
        seq_gc = seq.replace("A", "").replace("T", "")
        return float(len(seq_gc)) / len(seq)

    faidx = pyfaidx.Fasta(fasta, build_index=False, rebuild=False)
    tmp = concordance.apply(lambda x: _get_gc(x, window_size, faidx), axis=1, result_type="reduce")
    concordance["gc_content"] = list(x for x in list(tmp))
    return concordance


def get_coverage(
    df: pd.DataFrame,
    bw_coverage_files_high_quality: list[str],
    bw_coverage_files_low_quality: list[str],
) -> pd.DataFrame:
    """Adds coverage columns to the variant dataframe. Three columns are added: coverage - total coverage,
    well_mapped_coverage - coverage of reads with mapping quality > min_quality and repetitive_read_coverage -
    which is the difference between the two.
    bw_coverage_files should be outputs of `coverage_analysis.py` and have .chr??. inside the name

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (VCF or concordance)
    bw_coverage_files_high_quality : List[str]
        List of BW for the coverage at the high MAPQ quality threshold
    bw_coverage_files_low_quality : List[str]
        List of BW for the coverage at the MAPQ threshold 0


    Returns
    -------
    pd.DataFrame
        Modified dataframe
    """

    chrom2f = [(list(pbw.open(f).chroms().keys())[0], f) for f in bw_coverage_files_high_quality]
    chrom2bw_dict = dict(chrom2f)

    chrom2f = [(list(pbw.open(f).chroms().keys())[0], f) for f in bw_coverage_files_low_quality]
    for v_var in chrom2f:
        chrom2bw_dict[v_var[0]] = (chrom2bw_dict[v_var[0]], v_var[1])

    df.insert(len(df.columns), "coverage", np.NaN)
    df.insert(len(df.columns), "well_mapped_coverage", np.NaN)
    df.insert(len(df.columns), "repetitive_read_coverage", np.NaN)

    gdf = df.groupby("chrom")

    for g_var in gdf.groups:
        if g_var in chrom2bw_dict:
            bw_hq = chrom2bw_dict[g_var][0]
            bw_lq = chrom2bw_dict[g_var][1]
        else:
            continue

        with pbw.open(bw_hq) as bw_file:
            values_hq = [bw_file.values(g_var, x[1] - 1, x[1])[0] for x in gdf.groups[g_var]]
        with pbw.open(bw_lq) as bw_file:
            values_lq = [bw_file.values(g_var, x[1] - 1, x[1])[0] for x in gdf.groups[g_var]]
        df.loc[gdf.groups[g_var], "coverage"] = values_lq
        df.loc[gdf.groups[g_var], "well_mapped_coverage"] = values_hq
        df.loc[gdf.groups[g_var], "repetitive_read_coverage"] = np.array(values_lq) - np.array(values_hq)
    return df


def close_to_hmer_run(
    df: pd.DataFrame,
    runfile: str,
    min_hmer_run_length: int = 10,
    max_distance: int = 10,
) -> pd.DataFrame:
    """Adds column is_close_to_hmer_run and inside_hmer_run that is T/F"""
    df["close_to_hmer_run"] = False
    df["inside_hmer_run"] = False
    run_df = bed_writer.parse_intervals_file(runfile, min_hmer_run_length)
    gdf = df.groupby("chrom")
    grun_df = run_df.groupby("chromosome")
    for chrom in gdf.groups.keys():
        gdf_ix = gdf.groups[chrom]
        grun_ix = grun_df.groups[chrom]
        pos1 = np.array(df.loc[gdf_ix, "pos"])
        pos2 = np.array(run_df.loc[grun_ix, "start"])
        pos1_closest_pos2_start = np.searchsorted(pos2, pos1) - 1
        close_dist = abs(pos1 - pos2[np.clip(pos1_closest_pos2_start, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(pos1_closest_pos2_start + 1, None, len(pos2) - 1)] - pos1) < max_distance
        pos2 = np.array(run_df.loc[grun_ix, "end"])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)
        close_dist |= abs(pos1 - pos2[np.clip(pos1_closest_pos2_end - 1, 0, None)]) < max_distance
        close_dist |= abs(pos2[np.clip(pos1_closest_pos2_end, None, len(pos2) - 1)] - pos1) < max_distance
        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, "inside_hmer_run"] = is_inside
        df.loc[gdf_ix, "close_to_hmer_run"] = close_dist & (~is_inside)
    return df


def annotate_intervals(df: pd.DataFrame, annotfile: str) -> pd.DataFrame:
    """
    Adds column based on interval annotation file (T/F)

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be annotated
    annotfile: str
        bed file of the annotated intervals

    Returns
    -------
    pd.DataFrame
        Adds boolean column for the annotation (parsed from the file name)
    str
        Annotation name


    """
    annot = annotfile.split("/")[-1]
    if annot[-4:] == ".bed":
        annot = annot[:-4]

    df[annot] = False
    annot_df = bed_writer.parse_intervals_file(annotfile)
    gdf = df.groupby("chrom")
    gannot_df = annot_df.groupby("chromosome")
    for chrom in gdf.groups.keys():
        if chrom not in gannot_df.groups:
            continue
        gdf_ix = gdf.groups[chrom]
        gannot_ix = gannot_df.groups[chrom]
        pos1 = np.array(df.loc[gdf_ix, "pos"])
        pos2 = np.array(annot_df.loc[gannot_ix, "start"])
        pos1_closest_pos2_start = np.searchsorted(pos2, pos1) - 1
        pos2 = np.array(annot_df.loc[gannot_ix, "end"])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)

        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, annot] = is_inside
    return df, annot


def fill_filter_column(df: pd.DataFrame) -> pd.DataFrame:
    """Fills filter status column with PASS when there are missing values
    (e.g. when the FILTER column in the vcf has dots, or when false negative variants
    were added to the dataframe)

    Parameters
    ----------
    df : pd.DataFrame
        Description

    Returns
    -------
    pd.DataFrame
        Description
    """
    if "filter" not in df.columns:
        df["filter"] = np.nan
    fill_column_locs = (pd.isnull(df["filter"])) | (df["filter"] == "")
    df.loc[fill_column_locs, "filter"] = "PASS"
    return df


def annotate_cycle_skip(df: pd.DataFrame, flow_order: str, gt_field: str = None) -> pd.DataFrame:
    """Adds cycle skip information: non-skip, NA, cycle-skip, possible cycle-skip

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    flow_order : str
        Flow order
    gt_field: str
        If snps that correspond to a specific genotype are to be considered

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "cycleskip_status" - possible values are:
        * non-skip (not a cycle skip at any flow order)
        * NA (undefined - non-snps, multiallelic snps )
        * cycle-skip
        ( possible cycle-skip (cycle skip at a different flow order))
    """

    def is_multiallelic(x, gt_field):
        s_var = set(x[gt_field]) | set([0])
        if len(s_var) > 2:
            return True
        return False

    def get_non_ref(x):
        return [y for y in x if y != 0][0]

    def is_non_polymorphic(x, gt_field):
        s_var = set(x[gt_field]) | set([0])
        return len(s_var) == 1

    if gt_field is None:
        na_pos = df["indel"] | (df["alleles"].apply(len) > 2)
    else:
        na_pos = (
            df["indel"]
            | df.apply(lambda x: is_multiallelic(x, gt_field), axis=1)
            | df.apply(lambda x: is_non_polymorphic(x, gt_field), axis=1)
        )
    df["cycleskip_status"] = UNDETERMINED
    snp_pos = ~na_pos
    snps = df.loc[snp_pos].copy()
    left_last = np.array(snps["left_motif"]).astype(np.string_)
    right_first = np.array(snps["right_motif"]).astype(np.string_)

    ref = np.array(snps["ref"]).astype(np.string_)
    if gt_field is None:
        alt = np.array(snps["alleles"].apply(lambda x: x[1] if len(x) > 1 else None)).astype(np.string_)
    else:
        nra = snps[gt_field].apply(get_non_ref)
        snps["nra_idx"] = nra
        snps.loc[snps.nra_idx.isnull(), "nra_idx"] = 1
        snps["nra_idx"] = snps["nra_idx"].astype(np.int)
        alt = np.array(snps.apply(lambda x: x["alleles"][x["nra_idx"]], axis=1)).astype(np.string_)
        snps.drop("nra_idx", axis=1, inplace=True)

    ref_seqs = np.char.add(np.char.add(left_last, ref), right_first)
    alt_seqs = np.char.add(np.char.add(left_last, alt), right_first)

    ref_encs = [
        utils.catch(
            flowBasedRead.generate_key_from_sequence,
            str(np.char.decode(x)),
            flow_order,
            exception_type=ValueError,
            handle=lambda x: UNDETERMINED,
        )
        for x in ref_seqs
    ]
    alt_encs = [
        utils.catch(
            flowBasedRead.generate_key_from_sequence,
            str(np.char.decode(x)),
            flow_order,
            exception_type=ValueError,
            handle=lambda x: UNDETERMINED,
        )
        for x in alt_seqs
    ]

    cycleskip = np.array(
        [
            x
            for x in range(len(ref_encs))
            if isinstance(ref_encs[x], np.ndarray)
            and isinstance(alt_encs[x], np.ndarray)
            and len(ref_encs[x]) != len(alt_encs[x])
        ]
    )
    poss_cycleskip = [
        x
        for x in range(len(ref_encs))
        if isinstance(ref_encs[x], np.ndarray)
        and isinstance(alt_encs[x], np.ndarray)
        and len(ref_encs[x]) == len(alt_encs[x])
        and (
            np.any(ref_encs[x][ref_encs[x] - alt_encs[x] != 0] == 0)
            or np.any(alt_encs[x][ref_encs[x] - alt_encs[x] != 0] == 0)
        )
    ]

    undetermined = np.array(
        [
            x
            for x in range(len(ref_encs))
            if (isinstance(ref_encs[x], str) and ref_encs[x] == UNDETERMINED)
            or (isinstance(alt_encs[x], str) and alt_encs[x] == UNDETERMINED)
        ]
    )
    s_var = set(np.concatenate((cycleskip, poss_cycleskip, undetermined)))
    non_cycleskip = [x for x in range(len(ref_encs)) if x not in s_var]

    vals = [""] * len(snps)
    for x in cycleskip:
        vals[x] = "cycle-skip"
    for x in poss_cycleskip:
        vals[x] = "possible-cycle-skip"
    for x in non_cycleskip:
        vals[x] = "non-skip"
    for x in undetermined:
        vals[x] = UNDETERMINED
    snps["cycleskip_status"] = vals

    df.loc[snp_pos, "cycleskip_status"] = snps["cycleskip_status"]
    return df


def get_cycle_skip_dataframe(flow_order: str):
    """
    Generate a dataframe with the cycle skip status of all possible SNPs. The resulting dataframe is 192 rows, with the
    multi-index "ref_motif" and "alt_motif", which are the ref and alt of a SNP with 1 flanking base on each side
    (trinucleotide context). This output can be readily joined with a vcf dataframe once the right columns are created.

    Parameters
    ----------
    flow_order

    Returns
    -------

    """
    # build index composed of all possible SNPs
    ind = pd.MultiIndex.from_tuples(
        [
            x
            for x in itertools.product(
                ["".join(x) for x in itertools.product(["A", "C", "G", "T"], repeat=3)],
                ["A", "C", "G", "T"],
            )
            if x[0][1] != x[1]
        ],
        names=["ref_motif", "alt_motif"],
    )
    df_cskp = pd.DataFrame(index=ind).reset_index()
    df_cskp.loc[:, "alt_motif"] = (
        df_cskp["ref_motif"].str.slice(0, 1) + df_cskp["alt_motif"] + df_cskp["ref_motif"].str.slice(-1)
    )
    df_cskp.loc[:, CYCLE_SKIP_STATUS] = df_cskp.apply(
        lambda row: determine_cycle_skip_status(row["ref_motif"], row["alt_motif"], flow_order),
        axis=1,
    ).astype(CYCLE_SKIP_DTYPE)
    df_cskp.loc[:, IS_CYCLE_SKIP] = df_cskp[CYCLE_SKIP_STATUS] == CYCLE_SKIP
    return df_cskp.set_index(["ref_motif", "alt_motif"]).sort_index()


def determine_cycle_skip_status(ref: str, alt: str, flow_order: str):
    """return the cycle skip status, expects input of ref and alt sequences composed of 3 bases where only the 2nd base
    differs"""
    if len(ref) != 3 or len(alt) != 3 or ref[0] != alt[0] or ref[2] != alt[2] or ref == alt:
        raise ValueError(
            f"""Invalid inputs ref={ref}, alt={alt}
expecting input of ref and alt sequences composed of 3 bases where only the 2nd base differs"""
        )
    try:
        ref_key = np.trim_zeros(flowBasedRead.generate_key_from_sequence(ref, flow_order), "f")
        alt_key = np.trim_zeros(flowBasedRead.generate_key_from_sequence(alt, flow_order), "f")
        if len(ref_key) != len(alt_key):
            return CYCLE_SKIP

        for r_val, a_val in zip(ref_key, alt_key):
            if (r_val != a_val) and ((r_val == 0) or (a_val == 0)):
                return POSSIBLE_CYCLE_SKIP
        return NON_CYCLE_SKIP
    except ValueError:
        return UNDETERMINED_CYCLE_SKIP


def get_trinuc_sub_list():
    trinuc_list = ["".join([a, b, c]) for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    trinuc_sub = [
        ">".join([a, b]) for a in trinuc_list for b in trinuc_list if (a[0] == b[0]) & (a[2] == b[2]) & (a[1] != b[1])
    ]
    return trinuc_sub


def parse_trinuc_sub(rec):
    # X_LM and X_RM are the left and right motif sequences, respectively
    # obtained from AnnotateVcf.wdl pipeline
    motif_ref = rec.info["X_LM"][0][-1] + rec.ref + rec.info["X_RM"][0][0]
    motif_alt = rec.info["X_LM"][0][-1] + rec.alts[0] + rec.info["X_RM"][0][0]
    return motif_ref + ">" + motif_alt


def get_trinuc_substitution_dist(vcf_file):
    trinuc_sub = get_trinuc_sub_list()
    trinuc_dict = {k: 0 for k in trinuc_sub}
    with pysam.VariantFile(vcf_file, "rb") as infh:
        for rec in infh.fetch():
            # check if rec is a SNP
            if len(rec.ref) == 1 and len(rec.alts) == 1 and len(rec.alts[0]) == 1:
                trinuc_sub = parse_trinuc_sub(rec)
                trinuc_dict[trinuc_sub] += 1
    return trinuc_dict
