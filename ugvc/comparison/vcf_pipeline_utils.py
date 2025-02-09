"""Summary
"""
from __future__ import annotations

import os.path
import shutil
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import pyfaidx
import pysam
from simppl.simple_pipeline import SimplePipeline

import ugvc.comparison.flow_based_concordance as fbc
import ugbio_core.vcfbed.variant_annotation as annotation
from ugvc import logger
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.exec_utils import print_and_execute
from ugvc.vcfbed import vcftools
from ugbio_core.vcfbed import bed_writer


class VcfPipelineUtils:
    """Utilities of vcf pipeline, mostly wrappers around shell scripts

    Attributes
    ----------
    sp : SimplePipeline
        Simple pipeline object
    """

    def __init__(self, simple_pipeline: SimplePipeline | None = None):
        """Combines VCF in parts from GATK and indices the result

        Parameters
        ----------
        simple_pipeline : SimplePipeline, optional
            Optional SimplePipeline object for executing shell commands
        """
        self.sp = simple_pipeline

    def __execute(self, command: str, output_file: str | None = None):
        """Summary

        Parameters
        ----------
        command : str
            Description
        output_file : str, optional
            Description
        """
        print_and_execute(command, output_file=output_file, simple_pipeline=self.sp, module_name=__name__)

    def combine_vcf(self, n_parts: int, input_prefix: str, output_fname: str):
        """Combines VCF in parts from GATK and indices the result

        Parameters
        ----------
        n_parts : int
            Number of VCF parts (names will be 1-based)
        input_prefix : str
            Prefix of the VCF files (including directory) 1.vcf.gz ... will be added
        output_fname : str
            Name of the output VCF
        """
        input_files = [f"{input_prefix}.{x}.vcf" for x in range(1, n_parts + 1)] + [
            f"{input_prefix}.{x}.vcf.gz" for x in range(1, n_parts + 1)
        ]
        input_files = [x for x in input_files if os.path.exists(x)]
        self.__execute(f"bcftools concat -o {output_fname} -O z {input_files}")
        self.index_vcf(output_fname)

    def index_vcf(self, vcf: str):
        """Tabix index on VCF

        Parameters
        ----------
        vcf : str
            Input vcf.gz file
        """
        self.__execute(f"bcftools index -tf {vcf}")

    def reheader_vcf(self, input_file: str, new_header: str, output_file: str):
        """Run bcftools reheader and index

        Parameters
        ----------
        input_file : str
            Input file name
        new_header : str
            Name of the new header
        output_file : str
            Name of the output file

        No Longer Returned
        ------------------
        None, generates `output_file`
        """
        self.__execute(f"bcftools reheader -h {new_header} {input_file}")
        self.index_vcf(output_file)

    def intersect_bed_files(self, input_bed1: str, input_bed2: str, bed_output: str) -> None:
        """Intersects bed files

        Parameters
        ----------
        input_bed1 : str
            Input Bed file
        input_bed2 : str
            Input Bed file
        bed_output : str
            Output bed intersected file

        Writes output_fn file
        """
        self.__execute(f"bedtools intersect -a {input_bed1} -b {input_bed2}", output_file=bed_output)

    def run_vcfeval(
        self,
        vcf: str,
        gt: str,
        hcr: str,
        outdir: str,
        ref_genome: str,
        ev_region: str | None = None,
        output_mode: str = "split",
        samples: str | None = None,
        erase_outdir: bool = True,
        additional_args: str = "",
        score: str = "QUAL",
        all_records: bool = False,
    ):  # pylint: disable=too-many-arguments
        """
        Run vcfeval to evaluate the concordance between two VCF files
        Parameters
        ----------
        vcf : str
            Our variant calls
        gt : str
            GIAB (or other source truth file)
        hcr : str
            High confidence regions
        outdir : str
            Output directory
        ref_genome : str
            SDF reference file
        ev_region: str, optional
            Bed file of regions to evaluate (--bed-region)
        output_mode: str, optional
            Mode of vcfeval (default - split)
        samples: str, optional
            Sample names to compare (baseline,calls)
        erase_outdir: bool, optional
            Erase the output directory if it exists before running (otherwise vcfeval crashes)
        additional_args: str, optional
            Additional arguments to pass to vcfeval
        score: str, optional
            Score field to use for producing ROC curves in VCFEVAL
        all_records: bool, optional
            Include all records in the evaluation (default - False)
        """
        if erase_outdir and os.path.exists(outdir):
            shutil.rmtree(outdir)
        cmd = [
            "rtg",
            "RTG_MEM=12G",
            "vcfeval",
            "-b",
            gt,
            "-c",
            vcf,
            "-e",
            hcr,
            "-t",
            ref_genome,
            "-m",
            output_mode,
            "--decompose",
            "-f",
            score,
            "-o",
            outdir,
        ]
        if ev_region:
            cmd += ["--bed-regions", ev_region]
        if all_records:
            cmd += ["--all-records"]
        if additional_args:
            cmd += additional_args.split()
        if samples:
            cmd += ["--sample", samples]

        logger.info(" ".join(cmd))
        return self.__execute(" ".join(cmd))

    def intersect_with_intervals(self, input_fn: str, intervals_fn: str, output_fn: str) -> None:
        """Intersects VCF with intervalList. Writes output_fn file

        Parameters
        ----------
        input_fn : str
            Input file
        intervals_fn : str
            Interval_list file
        output_fn : str
            Output file

        Writes output_fn file
        """
        self.__execute(f"gatk SelectVariants -V {input_fn} -L {intervals_fn} -O {output_fn}")

    # pylint: disable=too-many-arguments
    def run_vcfeval_concordance(
        self,
        input_file: str,
        truth_file: str,
        output_prefix: str,
        ref_genome: str,
        evaluation_regions: str,
        comparison_intervals: str | None = None,
        input_sample: str | None = None,
        truth_sample: str | None = None,
        ignore_filter: bool = False,
        mode: str = "combine",
        ignore_genotype: bool = False,
    ) -> str:
        """Run vcfeval to evaluate concordance

        Parameters
        ----------
        input_file : str
            Our variant calls
        truth_file : str
            GIAB (or other source truth file)
        output_prefix : str
            Output prefix
        ref_genome : str
            Fasta reference file
        evaluation_regions: str
            Bed file of regions to evaluate (HCR)
        comparison_intervals: Optional[str]
            Picard intervals file to make the comparisons on. Default: None = all genome
        input_sample : str, optional
            Name of the sample in our input_file
        truth_sample : str, optional
            Name of the sample in the truth file
        ignore_filter : bool, optional
            Ignore status of the variant filter
        mode: str, optional
            Mode of vcfeval (default - combine)
        ignore_genotype: bool, optional
            Don't compare genotype information, only compare if allele is present in ground-truth
        Returns
        -------
        final concordance vcf file if the mode is "combine"
        otherwise - returns the output directory of vcfeval
        """

        output_dir = os.path.dirname(output_prefix)
        SDF_path = ref_genome + ".sdf"
        vcfeval_output_dir = os.path.join(output_dir, os.path.basename(output_prefix) + ".vcfeval_output")

        if os.path.isdir(vcfeval_output_dir):
            shutil.rmtree(vcfeval_output_dir)

        # filter the vcf to be only in the comparison_intervals.
        filtered_truth_file = os.path.join(output_dir, ".".join((os.path.basename(truth_file), "filtered", "vcf.gz")))
        if comparison_intervals is not None:
            self.intersect_with_intervals(truth_file, comparison_intervals, filtered_truth_file)
        else:
            shutil.copy(truth_file, filtered_truth_file)
            self.index_vcf(filtered_truth_file)

        if truth_sample is not None and input_sample is not None:
            samples = f"{truth_sample},{input_sample}"
        else:
            samples = None

        self.run_vcfeval(
            input_file,
            filtered_truth_file,
            evaluation_regions,
            vcfeval_output_dir,
            SDF_path,
            output_mode=mode,
            samples=samples,
            erase_outdir=True,
            additional_args="--squash_ploidy" if ignore_genotype else "",
            all_records=ignore_filter,
        )

        if mode == "combine":
            # fix the vcf file format
            self.fix_vcf_format(os.path.join(vcfeval_output_dir, "output"))

            # make the vcfeval output file without weird variants
            self.normalize_vcfeval_vcf(
                os.path.join(vcfeval_output_dir, "output.vcf.gz"),
                os.path.join(vcfeval_output_dir, "output.norm.vcf.gz"),
                ref_genome,
            )

            vcf_concordance_file = f'{output_prefix + ".vcfeval_concordance.vcf.gz"}'
            # move the file to be compatible with the output file of the genotype
            # concordance
            self.__execute(f'mv {os.path.join(vcfeval_output_dir, "output.norm.vcf.gz")} {vcf_concordance_file}')

            # generate index file for the vcf.gz file
            self.index_vcf(vcf_concordance_file)
            return vcf_concordance_file
        return vcfeval_output_dir

    def normalize_vcfeval_vcf(self, input_vcf: str, output_vcf: str, ref: str) -> None:
        """Combines monoallelic rows from VCFEVAL into multiallelic
        and combines the BASE/CALL annotations together. Mostly uses `bcftools norm`,
        but since it does not aggregate the INFO tags, they are aggregated using
        `bcftools annotate`.

        Parameters
        ----------
        input_vcf: str
            Input (output.vcf.gz from VCFEVAL)
        output_vcf: str
            Input (output.vcf.gz from VCFEVAL)
        ref: str
            Reference FASTA

        Returns
        -------
        None:
            Creates output_vcf
        """

        tempdir = f"{output_vcf}_tmp"
        os.mkdir(tempdir)

        # Step1 - bcftools norm

        self.__execute(f"bcftools norm -f {ref} -m+any -o {tempdir}/step1.vcf.gz -O z {input_vcf}")
        self.index_vcf(f"{tempdir}/step1.vcf.gz")
        self.__execute(
            f"bcftools annotate -a {input_vcf} -c CHROM,POS,CALL,BASE -Oz \
                -o {tempdir}/step2.vcf.gz {tempdir}/step1.vcf.gz"
        )
        self.index_vcf(f"{tempdir}/step2.vcf.gz")

        # Step2 - write out the annotation table. We use VariantsToTable from gatk, but remove
        # the mandatory header and replace NA by "."
        self.__execute(
            f"gatk VariantsToTable -V {input_vcf} -O {tempdir}/source.tsv -F CHROM -F POS -F REF -F CALL -F BASE"
        )
        self.__execute(
            f"gatk VariantsToTable -V {tempdir}/step2.vcf.gz -O {tempdir}/dest.tsv \
                -F CHROM -F POS -F REF -F CALL -F BASE"
        )

        # Step3 - identify lines that still need to be filled (where the CALL/BASE after the nomrmalization
        # are different from those that are expected from collapsing). This happens because plain bcftools annotate
        # requires match of reference allele that can change with normalization
        df1 = pd.read_csv(os.path.join(tempdir, "source.tsv"), sep="\t").set_index(["CHROM", "POS", "CALL"])
        df2 = pd.read_csv(os.path.join(tempdir, "dest.tsv"), sep="\t").set_index(["CHROM", "POS", "CALL"])

        difflines = df1.loc[df1.index.difference(df2.index)].reset_index().fillna(".")[["CHROM", "POS", "CALL"]]
        difflines.to_csv(os.path.join(tempdir, "step3.call.tsv"), sep="\t", header=False, index=False)

        df1 = pd.read_csv(os.path.join(tempdir, "source.tsv"), sep="\t").set_index(["CHROM", "POS", "BASE"])
        df2 = pd.read_csv(os.path.join(tempdir, "dest.tsv"), sep="\t").set_index(["CHROM", "POS", "BASE"])

        difflines = df1.loc[df1.index.difference(df2.index)].reset_index().fillna(".")[["CHROM", "POS", "BASE"]]
        difflines.to_csv(os.path.join(tempdir, "step3.base.tsv"), sep="\t", header=False, index=False)

        # Step 4 - annoate with the additional tsvs
        self.__execute(f"bgzip {tempdir}/step3.call.tsv")
        self.__execute(f"bgzip {tempdir}/step3.base.tsv")
        self.__execute(f"tabix -s1 -e2 -b2 {tempdir}/step3.call.tsv.gz")
        self.__execute(f"tabix -s1 -e2 -b2 {tempdir}/step3.base.tsv.gz")
        self.__execute(
            f"bcftools annotate -c CHROM,POS,CALL -a {tempdir}/step3.call.tsv.gz \
                -Oz -o {tempdir}/step4.vcf.gz {tempdir}/step2.vcf.gz"
        )
        self.index_vcf(f"{tempdir}/step4.vcf.gz")
        self.__execute(
            f"bcftools annotate -c CHROM,POS,CALL -a {tempdir}/step3.base.tsv.gz \
                -Oz -o {output_vcf} {tempdir}/step4.vcf.gz"
        )
        self.index_vcf(output_vcf)
        # shutil.rmtree(tempdir)

    def fix_vcf_format(self, output_prefix: str):
        """Legacy function to fix the PS field format in the old GIAB truth sets. The function overwrites the input file

        Parameters
        ----------
        output_prefix : str
            Prefix of the input and the output file (without the .vcf.gz)
        """
        self.__execute(f"gunzip -f {output_prefix}.vcf.gz")
        with open(f"{output_prefix}.vcf", encoding="utf-8") as input_file_handle:
            with open(f"{output_prefix}.tmp", "w", encoding="utf-8") as output_file_handle:
                for line in input_file_handle:
                    if line.startswith("##FORMAT=<ID=PS"):
                        output_file_handle.write(line.replace("Type=Integer", "Type=String"))
                    else:
                        output_file_handle.write(line)
        self.__execute(f"mv {output_file_handle.name} {input_file_handle.name}")
        self.__execute(f"bgzip {input_file_handle.name}")
        self.index_vcf(f"{input_file_handle.name}.gz")

    def annotate_tandem_repeats(self, input_file: str, reference_fasta: str) -> str:
        """Runs VariantAnnotator on the input file to add tandem repeat annotations (maybe others)

        Parameters
        ----------
        input_file : str
            vcf.gz file
        reference_fasta : str
            Reference file (should have .dict file nearby)

        Creates a copy of the input_file with .annotated.vcf.gz and the index
        Returns
        -------
        path to output file: str
        """

        output_file = input_file.replace("vcf.gz", "annotated.vcf.gz")
        self.__execute(f"gatk VariantAnnotator -V {input_file} -O {output_file} -R {reference_fasta} -A TandemRepeat")
        return output_file

    def transform_hom_calls_to_het_calls(self, input_file_calls: str, output_file_calls: str) -> None:
        """Reverse homozygous reference calls in deepVariant to filtered heterozygous so that max recall can be
        calculated

        Parameters
        ----------
        input_file_calls : str
            Input file name
        output_file_calls : str
            Output file name
        """

        with pysam.VariantFile(input_file_calls) as input_file:
            with pysam.VariantFile(output_file_calls, "w", header=input_file.header) as output_file:
                for rec in input_file:
                    if (
                        rec.samples[0]["GT"] == (0, 0)
                        or rec.samples[0]["GT"] == (None, None)
                        and "PASS" not in rec.filter
                    ):
                        rec.samples[0]["GT"] = (0, 1)
                    output_file.write(rec)
        self.index_vcf(output_file_calls)


def _fix_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Parses dataframe generated from the VCFEVAL concordance VCF and prepares it for
     classify/classify_gt functions that only consider the genotypes of the call and the base
    only rather than at the classification that VCFEVAL produced

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Output dataframe,
    """

    # remove genotypes of variants that were filtered out and thus are false negatives
    # (VCFEVAL outputs UG genotype for ignored genotypes too and they are classified downstream
    # as true positives if we do not make this modification)
    fix_tp_fn_loc = (df["call"] == "IGN") & ((df["base"] == "FN") | (df["base"] == "FN_CA"))
    replace = df.loc[fix_tp_fn_loc, "gt_ultima"].apply(lambda x: (None,))
    df.loc[replace.index, "gt_ultima"] = replace

    # fix all the places in which vcfeval returns a good result, but the genotype is not adequate
    # in these cases we change the genotype of the gt to be adequate with the classify function as follow:
    # (TP,TP), (TP,None) - should put the values of ultima in the gt
    df.loc[(df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna())), "gt_ground_truth"] = df[
        (df["call"] == "TP") & ((df["base"] == "TP") | (df["base"].isna()))
    ]["gt_ultima"]

    # (FP_CA,FN_CA), (FP_CA,None) - Fake a genotype from ultima such that one of the alleles is the same (and only one)
    df.loc[(df["call"] == "FP_CA") & ((df["base"] == "FN_CA") | (df["base"].isna())), "gt_ground_truth"] = df[
        (df["call"] == "FP_CA") & ((df["base"] == "FN_CA") | (df["base"].isna()))
    ]["gt_ultima"].apply(
        lambda x: ((x[0], x[0]) if (len(x) < 2 or (x[1] == 0)) else ((x[1], x[1]) if (x[0] == 0) else (x[0], 0)))
    )
    return df


def __map_variant_to_dict(variant: pysam.VariantRecord) -> defaultdict:
    """Converts a line from vcfeval concordance VCF to a dictionary. The following fields are extracted
    call genotype, base genotype, qual, chromosome, position, ref, alt and all values from the INFO column

    Parameters
    ----------
    variant : pysam.VariantRecord
        VCF record

    Returns
    -------
    defaultdict
        Output dictionary
    """
    call_sample_ind = 1
    gtr_sample_ind = 0

    return defaultdict(
        lambda: None,
        variant.info.items()
        + [
            ("GT_ULTIMA", variant.samples[call_sample_ind]["GT"]),
            ("GT_GROUND_TRUTH", variant.samples[gtr_sample_ind]["GT"]),
            ("QUAL", variant.qual),
            ("CHROM", variant.chrom),
            ("POS", variant.pos),
            ("REF", variant.ref),
            ("ALLELES", variant.alleles),
        ],
    )


def vcf2concordance(
    raw_calls_file: str,
    concordance_file: str,
    chromosome: str | None = None,
    scoring_field: str | None = None,
) -> pd.DataFrame:
    """Generates concordance dataframe

    Parameters
    ----------
    raw_calls_file : str
        File with GATK calls (.vcf.gz)
    concordance_file : str
        GenotypeConcordance/VCFEVAL output file (.vcf.gz)
    chromosome: str
        Fetch a specific chromosome (Default - all)
    scoring_field : str, optional
        The name of the INFO field that is used to score the variants.
        This value replaces the TREE_SCORE in the output data frame.
        When None TREE_SCORE is not replaced (default: None)

    No Longer Returned
    ------------------
    pd.DataFrame
    """

    if chromosome is None:
        concord_vcf = pysam.VariantFile(concordance_file)
    else:
        concord_vcf = pysam.VariantFile(concordance_file).fetch(chromosome)

    def call_filter(x):
        # Remove variants that were ignored (either outside of comparison intervals or
        # filtered out).
        return not (
            (x["CALL"] in {"IGN", "OUT"} and x["BASE"] is None)
            or (x["CALL"] in {"IGN", "OUT"} and x["BASE"] in {"IGN", "OUT"})
            or (x["CALL"] is None and x["BASE"] in {"IGN", "OUT"})
        )

    concord_vcf_extend = filter(call_filter, (__map_variant_to_dict(variant) for variant in concord_vcf))

    columns = [
        "CHROM",
        "POS",
        "QUAL",
        "REF",
        "ALLELES",
        "GT_ULTIMA",
        "GT_GROUND_TRUTH",
        "SYNC",
        "CALL",
        "BASE",
        "STR",
        "RU",
        "RPA",
    ]
    column_names = [x.lower() for x in columns]
    concordance_df = pd.DataFrame([[x[y] for y in columns] for x in concord_vcf_extend], columns=column_names)

    # make the gt_ground_truth compatible with GC
    concordance_df["gt_ground_truth"] = concordance_df["gt_ground_truth"].map(
        lambda x: (None, None) if x == (None,) else x
    )

    concordance_df["indel"] = concordance_df["alleles"].apply(lambda x: len({len(y) for y in x}) > 1)
    concordance_df = _fix_errors(concordance_df)

    def classify(x: defaultdict | pd.Series | dict) -> str:
        """Classify a record as true positive / false positive / false negative by matching the alleles.
        TP will be called if some of the alleles match

        Parameters
        ----------
        x : defaultdict, pd.Series or dict
            Input record

        Returns
        -------
        str
            classification
        """
        if x["gt_ultima"] == (None, None) or x["gt_ultima"] == (None,):
            return "fn"

        if x["gt_ground_truth"] == (None, None) or x["gt_ground_truth"] == (None,):
            return "fp"

        # If both gt_ultima and gt_ground_truth are not none:
        set_gtr = set(x["gt_ground_truth"]) - set([0])
        set_ultima = set(x["gt_ultima"]) - set([0])

        if len(set_gtr & set_ultima) > 0:
            return "tp"

        if len(set_ultima - set_gtr) > 0:
            return "fp"

        # If it is not tp or fp, then return fn:
        return "fn"

    concordance_df["classify"] = concordance_df.apply(classify, axis=1, result_type="reduce")

    def classify_gt(x: defaultdict | pd.Series | dict) -> str:
        """Classify a record as true positive / false negative / false positive. True positive requires
        match in the alleles and genotypes

        Parameters
        ----------
        x : defaultdict, pd.Series or dict
            Input record

        Returns
        -------
        str
            Classification
        """
        n_ref_gtr = len([y for y in x["gt_ground_truth"] if y == 0])
        n_ref_ultima = len([y for y in x["gt_ultima"] if y == 0])

        if x["gt_ultima"] == (None, None) or x["gt_ultima"] == (None,):
            return "fn"
        if x["gt_ground_truth"] == (None, None) or x["gt_ground_truth"] == (None,):
            return "fp"
        if n_ref_gtr < n_ref_ultima:
            return "fn"
        if n_ref_gtr > n_ref_ultima:
            return "fp"
        if x["gt_ultima"] != x["gt_ground_truth"]:
            return "fp"
        # If not fn or fp due to the reasons above:
        return "tp"

    concordance_df["classify_gt"] = concordance_df.apply(classify_gt, axis=1, result_type="reduce")
    concordance_df.loc[
        (concordance_df["classify_gt"] == "tp") & (concordance_df["classify"] == "fp"),
        "classify_gt",
    ] = "fp"

    # cases where we called wrong allele and then filtered out - are false negatives, not false positives
    called_fn = (concordance_df["base"] == "FN") | (concordance_df["base"] == "FN_CA")
    marked_fp = concordance_df["classify"] == "fp"
    concordance_df.loc[called_fn & marked_fp, "classify"] = "fn"
    marked_fp = concordance_df["classify_gt"] == "fp"
    concordance_df.loc[called_fn & marked_fp, "classify_gt"] = "fn"
    concordance_df.index = pd.Index(list(zip(concordance_df.chrom, concordance_df.pos)))
    original = vcftools.get_vcf_df(raw_calls_file, chromosome=chromosome, scoring_field=scoring_field)

    concordance_df.drop("qual", axis=1, inplace=True)

    drop_candidates = ["chrom", "pos", "alleles", "indel", "ref", "str", "ru", "rpa"]
    if original.shape[0] > 0:
        concordance = concordance_df.join(
            original.drop(
                [x for x in drop_candidates if x in original.columns and x in concordance_df.columns],
                axis=1,
            )
        )
    else:
        concordance = concordance_df.copy()

        tmp = original.drop(
            [x for x in drop_candidates if x in original.columns and x in concordance_df.columns],
            axis=1,
        )
        for t in tmp.columns:
            concordance[t] = None

    only_ref = concordance["alleles"].apply(len) == 1
    concordance = concordance[~only_ref]

    # Marking as false negative the variants that appear in concordance but not in the
    # original VCF (even if they do show some genotype)
    missing_variants = concordance.index.difference(original.index)
    logger.info("Identified %i variants missing in the input VCF", len(missing_variants))
    missing_variants_non_fn = concordance.loc[missing_variants].query("classify!='fn'").index
    logger.warning(
        "Identified %i variants missing in the input VCF and not marked false negatives",
        len(missing_variants_non_fn),
    )
    concordance.loc[missing_variants_non_fn, "classify"] = "fn"
    concordance.loc[missing_variants_non_fn, "classify_gt"] = "fn"

    return concordance


def bed_file_length(input_bed: str) -> int:
    """Calc the number of bases in a bed file

    Parameters
    ----------
    input_bed : str
        Input Bed file

    Return
    ------
    int
        number of bases in a bed file
    """

    df = pd.read_csv(input_bed, sep="\t", header=None)
    df = df.iloc[:, [0, 1, 2]]
    df.columns = ["chr", "pos_start", "pos_end"]
    return np.sum(df["pos_end"] - df["pos_start"] + 1)

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

def annotate_concordance(
    df: pd.DataFrame,
    fasta: str,
    bw_high_quality: list[str] | None = None,
    bw_all_quality: list[str] | None = None,
    annotate_intervals: list[str] | None = None,
    runfile: str | None = None,
    flow_order: str | None = DEFAULT_FLOW_ORDER,
    hmer_run_length_dist: tuple = (10, 10),
) -> tuple[pd.DataFrame, list]:
    """Annotates concordance data with information about SNP/INDELs and motifs

    Parameters
    ----------
    df : pd.DataFrame
        Concordance dataframe
    fasta : str
        Indexed FASTA of the reference genome
    bw_high_quality : list[str], optional
        Coverage bigWig file from high mapq reads  (Optional)
    bw_all_quality : list[str], optional
        Coverage bigWig file from all mapq reads  (Optional)
    annotate_intervals : list[str], optional
        Interval files for annotation
    runfile : str, optional
        bed file with positions of hmer runs (in order to mark homopolymer runs)
    flow_order : str, optional
        Flow order
    hmer_run_length_dist : tuple, optional
        tuple (min_hmer_run_length, max_distance) for marking variants near homopolymer runs

    No Longer Returned
    ------------------
    pd.DataFrame
        Annotated dataframe
    list
        list of the names of the annotations
    """

    if annotate_intervals is None:
        annotate_intervals = []

    logger.info("Marking SNP/INDEL")
    df = annotation.classify_indel(df)
    logger.info("Marking H-INDEL")
    df = annotation.is_hmer_indel(df, fasta)
    logger.info("Marking motifs")
    df = annotation.get_motif_around(df, 5, fasta)
    logger.info("Marking GC content")
    df = annotation.get_gc_content(df, 10, fasta)
    if bw_all_quality is not None and bw_high_quality is not None:
        logger.info("Calculating coverage")
        df = annotation.get_coverage(df, bw_high_quality, bw_all_quality)
    if runfile is not None:
        length, dist = hmer_run_length_dist
        logger.info("Marking homopolymer runs")
        df = close_to_hmer_run(df, runfile, min_hmer_run_length=length, max_distance=dist)
    annots = []
    if annotate_intervals is not None:
        for annotation_file in annotate_intervals:
            logger.info("Annotating intervals")
            df, annot = annotation.annotate_intervals(df, annotation_file)
            annots.append(annot)
    logger.debug("Filling filter column")  # debug since not interesting step
    df = annotation.fill_filter_column(df)

    logger.info("Filling filter column")
    if flow_order is not None:
        df = annotation.annotate_cycle_skip(df, flow_order=flow_order)
    return df, annots


def reinterpret_variants(
    concordance_df: pd.DataFrame,
    reference_fasta: str,
    ignore_low_quality_fps: bool = False,
) -> pd.DataFrame:
    """Reinterprets the variants by comparing the variant to the ground truth in flow space

    Parameters
    ----------
    concordance_df : pd.DataFrame
        Input dataframe
    reference_fasta : str
        Indexed FASTA
    ignore_low_quality_fps : bool, optional
        Shoud the low quality false positives be ignored in reinterpretation (True for mutect, default False)

    See Also
    --------
    `flow_based_concordance.py`

    No Longer Returned
    ------------------
    pd.DataFrame
        Reinterpreted dataframe
    """
    logger.info("Variants reinterpret")
    concordance_df_result = pd.DataFrame()
    fasta = pyfaidx.Fasta(reference_fasta, build_index=False, rebuild=False)
    for contig in concordance_df["chrom"].unique():
        concordance_df_contig = concordance_df.loc[concordance_df["chrom"] == contig]
        input_dict = _get_locations_to_work_on(concordance_df_contig, ignore_low_quality_fps)
        concordance_df_contig = fbc.reinterpret_variants(concordance_df_contig, input_dict, fasta)
        concordance_df_result = pd.concat([concordance_df_result, concordance_df_contig])
    return concordance_df_result


def _get_locations_to_work_on(input_df: pd.DataFrame, ignore_low_quality_fps: bool = False) -> dict:
    """Dictionary of  in the dataframe that we care about

    Parameters
    ----------
    input_df : pd.DataFrame
        Input
    ignore_low_quality_fps : bool, optional
        Should we ignore the low quality false positives

    Returns
    -------
    dict
        locations dictionary split between fps/fns/tps etc.

    """
    df = vcftools.FilterWrapper(input_df)
    fps = df.reset().get_fp().get_df()
    if "tree_score" in fps.columns and fps["tree_score"].dtype == np.float64 and ignore_low_quality_fps:
        cutoff = fps.tree_score.quantile(0.80)
        fps = fps.query(f"tree_score > {cutoff}")
    fns = df.reset().get_df().query('classify=="fn"')
    tps = df.reset().get_tp().get_df()
    gtr = (
        df.reset().get_df().loc[df.get_df()["gt_ground_truth"].apply(lambda x: x not in [(None, None), (None,)])].copy()
    )
    gtr.sort_values("pos", inplace=True)
    ugi = df.reset().get_df().loc[df.get_df()["gt_ultima"].apply(lambda x: x not in [(None, None), (None,)])].copy()
    ugi.sort_values("pos", inplace=True)

    pos_fps = np.array(fps.pos)
    pos_gtr = np.array(gtr.pos)
    pos_ugi = np.array(ugi.pos)
    pos_fns = np.array(fns.pos)

    result = {
        "fps": fps,
        "fns": fns,
        "tps": tps,
        "gtr": gtr,
        "ugi": ugi,
        "pos_fps": pos_fps,
        "pos_gtr": pos_gtr,
        "pos_ugi": pos_ugi,
        "pos_fns": pos_fns,
    }

    return result
