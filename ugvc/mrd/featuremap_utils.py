from __future__ import annotations

import itertools
import os
from enum import Enum

import numpy as np
import pyfaidx
import pysam
from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugvc.dna.format import ALT, CHROM, DEFAULT_FLOW_ORDER, FILTER, IS_CYCLE_SKIP, POS, QUAL, REF
from ugvc.dna.utils import get_max_softclip_len
from ugvc.utils.exec_utils import print_and_execute
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json
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
    X_FC2 = "X_FC2"
    X_QUAL = "X_QUAL"
    X_RN = "X_RN"
    TRINUC_CONTEXT_WITH_ALT = "trinuc_context_with_alt"
    HMER_CONTEXT_REF = "hmer_context_ref"
    HMER_CONTEXT_ALT = "hmer_context_alt"
    IS_CYCLE_SKIP = IS_CYCLE_SKIP
    IS_FORWARD = "is_forward"
    IS_DUPLICATE = "is_duplicate"
    MAX_SOFTCLIP_LENGTH = "max_softclip_length"
    X_FLAGS = "X_FLAGS"
    X_CIGAR = "X_CIGAR"
    PREV_1 = "prev_1"
    PREV_2 = "prev_2"
    PREV_3 = "prev_3"
    NEXT_1 = "next_1"
    NEXT_2 = "next_2"
    NEXT_3 = "next_3"


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
        - trinuc_context_with_alt: reference trinucleotide context
        - prev_N: base i in the reference before the variant, i in range 1 to N=length motif_length_to_annotate
        - next_N: base i in the reference after the variant, i in range 1 to N=length motif_length_to_annotate
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
            The length of the motif to annotate context up to (prev_N / next_N). Defaults to 3.
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

        # info field names
        self.TRINUC_CONTEXT_WITH_ALT = FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value
        self.HMER_CONTEXT_REF = FeatureMapFields.HMER_CONTEXT_REF.value
        self.HMER_CONTEXT_ALT = FeatureMapFields.HMER_CONTEXT_ALT.value
        self.CYCLE_SKIP_FLAG = FeatureMapFields.IS_CYCLE_SKIP.value
        self.SINGLE_REF_BASES = []
        for i in range(self.motif_length_to_annotate):
            self.SINGLE_REF_BASES.append(f"prev_{i + 1}")
            self.SINGLE_REF_BASES.append(f"next_{i + 1}")

        self.info_fields_to_add = [
            self.TRINUC_CONTEXT_WITH_ALT,
            self.HMER_CONTEXT_REF,
            self.HMER_CONTEXT_ALT,
        ] + self.SINGLE_REF_BASES
        # self.CYCLE_SKIP_FLAG not included because it's a flag, only there if True

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

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.TRINUC_CONTEXT_WITH_ALT),
                ("Number", 1),
                ("Type", "String"),
                ("Description", "reference trinucleotide context and alt base"),
            ],
        )

        for i in range(self.motif_length_to_annotate):
            header.add_meta(
                "INFO",
                items=[
                    ("ID", f"prev_{i + 1}"),
                    ("Number", 1),
                    ("Type", "String"),
                    ("Description", f"{i + 1} bases in the reference before variant"),
                ],
            )
            header.add_meta(
                "INFO",
                items=[
                    ("ID", f"next_{i + 1}"),
                    ("Number", 1),
                    ("Type", "String"),
                    ("Description", f"{i + 1} bases in the reference after variant"),
                ],
            )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.HMER_CONTEXT_REF),
                ("Number", 1),
                ("Type", "Integer"),
                ("Description", f"reference homopolymer context, up to length {self.max_hmer_length}"),
            ],
        )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.HMER_CONTEXT_ALT),
                ("Number", 1),
                ("Type", "Integer"),
                (
                    "Description",
                    f"homopolymer context in the ref allele (assuming the variant considered only), "
                    f"up to length {self.max_hmer_length}",
                ),
            ],
        )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.CYCLE_SKIP_FLAG),
                ("Number", 0),
                ("Type", "Flag"),
                ("Description", "True if the SNV is a cycle skip"),
            ],
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
        faidx_ref = pyfaidx.Fasta(self.ref_fasta, build_index=False, rebuild=False)
        for j, record in enumerate(records):
            if is_biallelic_snv(record):
                # get motif
                ref_around_snv = get_motif_around_snv(record, self.max_hmer_length, faidx_ref)
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
                record.info[self.TRINUC_CONTEXT_WITH_ALT] = trinuc_ref + record.alts[0]

                for index in range(1, self.motif_length_to_annotate + 1):
                    field_name = f"next_{index}"
                    record.info[field_name] = ref_around_snv[central_base_ind + index]
                    field_name = f"prev_{index}"
                    record.info[field_name] = ref_around_snv[central_base_ind - index]

                is_cycle_skip = self.cycle_skip_dataframe.loc[(trinuc_ref, trinuc_alt), IS_CYCLE_SKIP]
                record.info[self.CYCLE_SKIP_FLAG] = is_cycle_skip

                # make sure all the info fields are present
                for info_field in self.info_fields_to_add:
                    assert (
                        info_field in record.info
                    ), f"INFO field {info_field} was supposed to be added to VCF record but was not found"
            records_out[j] = record

        return records_out


def create_hom_snv_featuremap(
    featuremap: str,
    sorter_stats_json: str = None,
    hom_snv_featuremap: str = None,
    sp: SimplePipeline = None,
    requested_min_coverage: int = 20,
    min_af: float = 0.7,
):
    """Create a HOM SNV featuremap from a featuremap

    Parameters
    ----------
    featuremap : str
        Input featuremap.
    sorter_stats_json : str
        Path to Sorter statistics JSON file, used to extract the median coverage. If None (default), minimum coverage
        will be set to requested_min_coverage even if the median coverage is lower, might yield an empty output.
    hom_snv_featuremap : str, optional
        Output featuremap with HOM SNVs reads to be used as True Positives. If None (default),
        the hom_snv_featuremap will be the same as the input featuremap with a ".hom_snv.vcf.gz" suffix.
    sp : SimplePipeline, optional
        SimplePipeline object to use for running commands. If None (default), commands will be run using subprocess.
    requested_min_coverage : int, optional
        Minimum coverage requested for locus to be propagated to the output. If the median coverage is lower than this
        value, the median coverage will be used as the minimum coverage instead.
        Default 20
    min_af : float, optional
        Minimum allele fraction in the featuremap to be considered a HOM SNV
        Default 0.7
        The default is chosen as 0.7 and not higher because some SNVs are pre-filtered from the FeatureMap due to
        MAPQ<60 or due to adjacent hmers.
    """

    # check inputs
    assert os.path.isfile(featuremap), f"featuremap {featuremap} does not exist"
    if sorter_stats_json:
        assert os.path.isfile(sorter_stats_json), f"sorter_stats_json {sorter_stats_json} does not exist"
    if hom_snv_featuremap is None:
        if featuremap.endswith(".vcf.gz"):
            hom_snv_featuremap = featuremap[: -len(".vcf.gz")]
        hom_snv_featuremap = featuremap + ".hom_snv.vcf.gz"
    hom_snv_bed = hom_snv_featuremap.replace(".vcf.gz", ".bed.gz")
    logger.info(f"Writting HOM SNV featuremap to {hom_snv_featuremap}")

    # get minimum coverage
    if sorter_stats_json:
        (
            _,
            _,
            _,
            min_coverage,
            _,
        ) = read_effective_coverage_from_sorter_json(sorter_stats_json, min_coverage_for_fp=requested_min_coverage)
    else:
        min_coverage = requested_min_coverage
    logger.info(
        f"Using a minimum coverage of {min_coverage} for HOM SNV featuremap (requested {requested_min_coverage})"
    )

    # Create commands to filter the featuremap for homozygous SNVs.
    cmd_get_hom_snv_loci_bed_file = (
        # Use bcftools to query specific fields in the vcf file. This includes the chromosome (CHROM),
        # the 0-based start position (POS0), the 1-based start position (POS), and the number of reads
        # in the locus (X_READ_COUNT) for the specified feature map.
        f"bcftools query -f '%CHROM\t%POS0\t%POS\t%INFO/{FeatureMapFields.READ_COUNT.value}\n' {featuremap} |"
        # Pipe the output to bedtools groupby command.
        # Here, -c 3 means we are specifying the third column as the key to groupby.
        # The '-full' option includes all columns from the input in the output.
        # The '-o count' option is specifying to count the number of lines for each group.
        f"bedtools groupby -c 3 -full -o count | "
        # Pipe the result to an awk command, which filters the result based on minimum coverage and allele frequency.
        # The '$4>=~{min_coverage}' part checks if the fourth column (which should be read count) is greater than or
        # equal to the minimum coverage. The '$5/$4>=~{min_af}' part checks if the allele frequency (calculated as
        # column 5 divided by column 4) is greater than or equal to the minimum allele frequency.
        f"awk '($4>={min_coverage})&&($5/$4>={min_af})' | "
        # The final output is then compressed and saved to the specified location in .bed.gz format.
        f"gzip > {hom_snv_bed}"
    )
    cmd_intersect_bed_file_with_original_featuremap = (
        f"bedtools intersect -a {featuremap} -b {hom_snv_bed} -u -header | bcftools view - -Oz -o {hom_snv_featuremap}"
    )
    cmd_index_hom_snv_featuremap = f"bcftools index -ft {hom_snv_featuremap}"

    # Run the commands
    try:
        for command in (
            cmd_get_hom_snv_loci_bed_file,
            cmd_intersect_bed_file_with_original_featuremap,
            cmd_index_hom_snv_featuremap,
        ):
            print_and_execute(command, simple_pipeline=sp, module_name=__name__)

    finally:
        # remove temp file
        if os.path.isfile(hom_snv_bed):
            os.remove(hom_snv_bed)


def filter_featuremap_with_bcftools_view(
    input_featuremap_vcf: str,
    intersect_featuremap_vcf: str,
    min_coverage: int = None,
    max_coverage: int = None,
    regions_file: str = None,
    bcftools_include_filter: str = None,
    sp: SimplePipeline = None,
) -> str:
    """
    Create a bcftools view command to filter a featuremap vcf

    Parameters
    ----------
    input_featuremap_vcf : str
        Path to input featuremap vcf
    intersect_featuremap_vcf : str
        Path to output intersected featuremap vcf
    min_coverage : int, optional
        Minimum coverage to include, by default None
    max_coverage : int, optional
        Maximum coverage to include, by default None
    regions_file : str, optional
        Path to regions file, by default None
    bcftools_include_filter: str, optional
        bcftools include filter to apply as part of a "bcftools view <vcf> -i 'pre_filter_bcftools_include'"
        before sampling, by default None
    sp : SimplePipeline, optional
        SimplePipeline object to use for printing and running commands, by default None
    """
    bcftools_view_command = f"bcftools view {input_featuremap_vcf} -O z -o {intersect_featuremap_vcf} "
    include_filters = [bcftools_include_filter.replace("'", '"')] if bcftools_include_filter else []
    # filter out variants with coverage outside of min and max coverage
    if min_coverage is not None:
        include_filters.append(f"(INFO/{FeatureMapFields.READ_COUNT.value} >= {min_coverage})")
    if max_coverage is not None:
        include_filters.append(f"(INFO/{FeatureMapFields.READ_COUNT.value} <= {max_coverage})")

    bcftools_include_string = " && ".join(include_filters)
    if bcftools_include_string:
        bcftools_view_command += f" -i '{bcftools_include_string}' "
    if regions_file:
        bcftools_view_command += f" -T {regions_file} "
    bcftools_index_command = f"bcftools index -t {intersect_featuremap_vcf}"

    print_and_execute(bcftools_view_command, simple_pipeline=sp, module_name=__name__, shell=True)
    print_and_execute(
        bcftools_index_command,
        simple_pipeline=sp,
        module_name=__name__,
    )
    assert os.path.isfile(intersect_featuremap_vcf), f"failed to create {intersect_featuremap_vcf}"
