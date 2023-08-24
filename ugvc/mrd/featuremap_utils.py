from __future__ import annotations

import itertools
import os
from enum import Enum

import numpy as np
import pyfaidx
import pysam

from ugvc.dna.format import ALT, CHROM, DEFAULT_FLOW_ORDER, FILTER, IS_CYCLE_SKIP, POS, QUAL, REF
from ugvc.dna.utils import get_max_softclip_len
from ugvc.vcfbed.variant_annotation import VcfAnnotator, get_cycle_skip_dataframe, get_motif_around_snv


class FeatureMapFields(Enum):
    CHROM = CHROM
    POS = POS
    REF = REF
    ALT = ALT
    QUAL = QUAL
    FILTER = FILTER
    READ_COUNT = "X_READ_COUNT"
    FILTERED_COUNT = "X_FILTERED_COUNT"
    X_SCORE = "X_SCORE"
    X_EDIST = "X_EDIST"
    X_LENGTH = "X_LENGTH"
    X_MAPQ = "X_MAPQ"
    X_INDEX = "X_INDEX"
    X_FC1 = "X_FC1"
    TRINUC_CONTEXT = "trinuc_context"
    HMER_CONTEXT_REF = "hmer_context_ref"
    HMER_CONTEXT_ALT = "hmer_context_alt"
    IS_CYCLE_SKIP = IS_CYCLE_SKIP
    IS_FORWARD = "is_forward"
    IS_DUPLICATE = "is_duplicate"
    MAX_SOFTCLIP_LENGTH = "max_softclip_length"
    X_FLAGS = "X_FLAGS"
    X_CIGAR = "X_CIGAR"
    PREV_3bp = "prev_3bp"
    NEXT_3bp = "next_3bp"


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


class FeaturemapAnnotator(VcfAnnotator):
    """
    Annotate vcf with featuremap-specific fields derived from X_FLAGS and X_CIGAR:
    - is_forward: is the read forward mapped
    - is_duplicate: is the read a duplicate
    - max_softclip_length: maximum softclip length
    """

    def __init__(self):
        pass

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
        header.add_line("##ugvc_ugvc/mrd/mrd_utils.py_FeaturemapAnnotator=.")
        header.add_line(
            f"##INFO=<ID={FeatureMapFields.IS_FORWARD.value},"
            'Number=0,Type=Flag,Description="is the read forward mapped">'
        )
        header.add_line(
            f"##INFO=<ID={FeatureMapFields.IS_DUPLICATE.value},"
            'Number=0,Type=Flag,Description="is the read a duplicate">'
        )
        header.add_line(
            f"##INFO=<ID={FeatureMapFields.MAX_SOFTCLIP_LENGTH.value},"
            'Number=1,Type=Integer,Description="maximum softclip length between start and end of read">'
        )

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
            if FeatureMapFields.X_FLAGS.value in record.info:
                flags = record.info[FeatureMapFields.X_FLAGS.value]
                record.info[FeatureMapFields.IS_FORWARD.value] = (flags & 16) == 0
                record.info[FeatureMapFields.IS_DUPLICATE.value] = (flags & 1024) != 0
            if FeatureMapFields.X_CIGAR.value in record.info:
                record.info[FeatureMapFields.MAX_SOFTCLIP_LENGTH.value] = get_max_softclip_len(
                    record.info[FeatureMapFields.X_CIGAR.value]
                )
            records_out[j] = record

        return records_out


# pylint: disable=too-many-instance-attributes
class RefContextVcfAnnotator(VcfAnnotator):
    def __init__(
        self,
        ref_fasta: str,
        flow_order: str = DEFAULT_FLOW_ORDER,
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
        self.TRINUC_CONTEXT = FeatureMapFields.TRINUC_CONTEXT.value
        self.HMER_CONTEXT_REF = FeatureMapFields.HMER_CONTEXT_REF.value
        self.HMER_CONTEXT_ALT = FeatureMapFields.HMER_CONTEXT_ALT.value
        self.CYCLE_SKIP_FLAG = FeatureMapFields.IS_CYCLE_SKIP.value
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
        header.add_line(
            f"##INFO=<ID={self.CYCLE_SKIP_FLAG}," f'Number=0,Type=Flag,Description="True if the SNV is a cycle skip">'
        )

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
