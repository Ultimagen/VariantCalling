from __future__ import annotations

import shutil
from os.path import basename, dirname
from os.path import join as pjoin
from os.path import splitext

from ugbio_comparison.vcf_pipeline_utils import VcfPipelineUtils
from ugvc.vcfbed.interval_file import IntervalFile


class ComparisonPipeline:  # pylint: disable=too-many-instance-attributes
    """
    Run comparison between the two sets of calls: input_prefix and truth_file. Creates
    a combined call file and a concordance VCF by either of the concordance tools
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        vpu: VcfPipelineUtils,
        n_parts: int,
        input_prefix: str,
        truth_file: str,
        cmp_intervals: IntervalFile,
        highconf_intervals: IntervalFile,
        ref_genome: str,
        call_sample: str,
        truth_sample: str,
        output_file_name: str,
        header: str | None = None,
        output_suffix: str | None = None,
        ignore_filter: bool = False,
        revert_hom_ref: bool = False,
    ):
        """
        Parameters
        ----------
        vpu: VcfPipelineUtils
            VcfPipelineUtils object for executing common functions via unix shell
        n_parts : int
            For input VCF split into number of parts - specifiy the number of parts. Specify
            zero for complete VCF
        input_prefix : str
            Input prefix for the vcf. If the vcf is split into multiple parts, the script
            will look for <input_prefix>.1.vcf, <input_prefix>.2.vcf etc. For the non-split VCF
            will look for <input_prefix>.vcf.gz
        truth_file : str, optional
            Truth calls file
        cmp_intervals : vcf_pipeline_utils.IntervalFile, optional
            interval_list file over which to do comparison (e.g. chr9)
        highconf_intervals : IntervalFile
            high confidence intervals for the ground truth
        ref_genome : str, optional
            Reference genome FASTA
        call_sample : str, optional
            Name of the calls sample
        truth_sample : str, optional
            Name of the truth sample
        output_file_name: str
            Name of the output file - will determine the name of the vcfeval output directory.
            The output_file_name should include the output_dir and output_dir and output_file_name
            are mutually exclusive
        header : str, optional
            for backward compatibility - to be able to change the header of the VCF. Default None
        output_suffix : str, optional
            Suffix for the output file name (e.g. chr9) -
            otherwise the output file nams are starting with the input prefix
        ignore_filter : bool, optional
            Should the filter status **of calls only** be ignored. Filter status of truth is always
            taken into account
        revert_hom_ref : bool, optional
            Should the hom ref calls filtered out be reverted (deepVariant only)
        """
        self.vpu = vpu
        self.n_parts = n_parts
        self.input_prefix = input_prefix
        self.truth_file = truth_file
        self.cmp_intervals = cmp_intervals
        self.ref_genome = ref_genome
        self.highconf_intervals = highconf_intervals
        self.call_sample = call_sample
        self.truth_sample = truth_sample
        self.output_file_name = output_file_name
        self.ignore_filter = ignore_filter
        self.revert_hom_ref = revert_hom_ref
        self.header = header
        self.output_suffix = output_suffix

        self.output_dir = dirname(output_file_name)
        self.output_prefix = splitext(output_file_name)[0]
        self.input_prefix_basename = basename(input_prefix)
        self.output_suffix = "" if not self.output_suffix else "." + self.output_suffix

    def run(self) -> tuple[str, str]:
        """
        Returns
        -------
        high_conf_calls_vcf, high_conf_concordance_vcf: Tuple[str, str]
        """

        combined_fn = self.__combine_vcf()
        reheader_fn = self.__reheader_vcf(combined_fn)
        revert_fn = self.__revert_hom_ref(reheader_fn)
        select_intervals_fn = self.__select_comparison_intervals(revert_fn)

        concordance_vcf = self.vpu.run_vcfeval_concordance(
            select_intervals_fn,
            self.truth_file,
            self.output_prefix,
            self.ref_genome,
            self.highconf_intervals.as_bed_file(),
            self.cmp_intervals.as_bed_file(),
            self.call_sample,
            self.truth_sample,
            self.ignore_filter,
            "combine",
        )
        annotated_concordance_vcf = self.vpu.annotate_tandem_repeats(concordance_vcf, self.ref_genome)

        high_conf_calls_vcf = select_intervals_fn.replace("vcf.gz", "highconf.vcf.gz")
        self.vpu.intersect_with_intervals(
            select_intervals_fn, self.highconf_intervals.as_interval_list_file(), high_conf_calls_vcf
        )

        high_conf_concordance_vcf = annotated_concordance_vcf.replace("vcf.gz", "highconf.vcf.gz")

        self.vpu.intersect_with_intervals(
            annotated_concordance_vcf, self.highconf_intervals.as_interval_list_file(), high_conf_concordance_vcf
        )
        return high_conf_calls_vcf, high_conf_concordance_vcf

    def __combine_vcf(self):
        output_fn = pjoin(self.output_dir, self.input_prefix_basename + f"{self.output_suffix}.vcf.gz")
        if self.n_parts > 0:
            self.vpu.combine_vcf(self.n_parts, self.input_prefix, output_fn)
        else:
            output_fn = self.input_prefix + ".vcf.gz"
        return output_fn

    def __reheader_vcf(self, output_fn):
        reheader_fn = pjoin(self.output_dir, self.input_prefix_basename + f"{self.output_suffix}.rhdr.vcf.gz")
        if self.header is not None:
            self.vpu.reheader_vcf(output_fn, self.header, reheader_fn)
        else:
            shutil.copy(output_fn, reheader_fn)
            shutil.copy(".".join((output_fn, "tbi")), ".".join((reheader_fn, "tbi")))
        return reheader_fn

    def __revert_hom_ref(self, reheader_fn):
        revert_fn = pjoin(self.output_dir, self.input_prefix_basename + f"{self.output_suffix}.rev.hom.ref.vcf.gz")
        if self.revert_hom_ref:
            self.vpu.transform_hom_calls_to_het_calls(reheader_fn, revert_fn)
        else:
            shutil.copy(reheader_fn, revert_fn)
            shutil.copy(".".join((reheader_fn, "tbi")), ".".join((revert_fn, "tbi")))
        return revert_fn

    def __select_comparison_intervals(self, revert_fn):
        select_intervals_fn = pjoin(self.output_dir, self.input_prefix_basename + f"{self.output_suffix}.intsct.vcf.gz")
        if not self.cmp_intervals.is_none():
            self.vpu.intersect_with_intervals(
                revert_fn, self.cmp_intervals.as_interval_list_file(), select_intervals_fn
            )
        else:
            shutil.copy(revert_fn, select_intervals_fn)
            self.vpu.index_vcf(select_intervals_fn)
        return select_intervals_fn
