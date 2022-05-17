import shutil
from os.path import basename, dirname
from os.path import join as pjoin
from os.path import splitext
from typing import Optional, Tuple

from simppl.simple_pipeline import SimplePipeline

import vcfbed.interval_file
from ugvc.comparison import vcf_pipeline_utils

CONCORDANCE_TOOL = "VCFEVAL"


def pipeline(
    sp: SimplePipeline,
    n_parts: int,
    input_prefix: str,
    truth_file: str,
    cmp_intervals: vcfbed.interval_file.IntervalFile,
    highconf_intervals: vcfbed.interval_file.IntervalFile,
    ref_genome: str,
    call_sample: str,
    truth_sample: str,
    output_dir: Optional[str] = None,
    output_file_name: Optional[str] = None,
    header: Optional[str] = None,
    runs_intervals: Optional[vcfbed.interval_file.IntervalFile] = None,
    output_suffix: Optional[str] = None,
    ignore_filter: bool = False,
    concordance_tool: str = CONCORDANCE_TOOL,
) -> Tuple[str, str]:
    """
    Run comparison between the two sets of calls: input_prefix and truth_file. Creates
    a combined call file and a concordance VCF by either of the concordance tools

    Parameters
    ----------
    sp: SimplePipeline
        SimplePipeline object for running shell commands conveniently
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
    highconf_intervals : str, optional
        high confidence intervals for the ground truth (BED)
    ref_genome : str, optional
        Reference genome FASTA
    call_sample : str, optional
        Name of the calls sample
    truth_sample : str, optional
        Name of the truth sample
    output_dir : str
        Location for the output
    output_file_name: str, optional
        Name of the output file - will determine the name of the vcfeval output directory.
        The output_file_name should include the output_dir and output_dir and output_file_name are mutually exclusive
    header : str, optional
        for backward compatibility - to be able to change the header of the VCF. Default None
    runs_intervals : str, optional
        Hompolymer runs annotation (BED)
    output_suffix : str, optional
        Suffix for the output file name (e.g. chr9) -
        otherwise the output file nams are starting with the input prefix
    ignore_filter : bool, optional
        Should the filter status **of calls only** be ignored. Filter status of truth is always
        taken into account
    concordance_tool : str, optional
        GC - GenotypeConcordance (picard) or VCFEVAL (default)

    Returns
    -------
    Tuple[str, str]
    """

    assert output_dir is None or output_file_name is None, ""
    if output_dir is None:
        if output_file_name is not None:
            output_dir = dirname(output_file_name)
        else:
            output_dir = dirname(input_prefix)

    input_prefix_basename = basename(input_prefix)

    if not output_suffix:
        output_fn = pjoin(output_dir, input_prefix_basename + ".vcf.gz")
    else:
        output_fn = pjoin(
            output_dir, input_prefix_basename + f".{output_suffix}.vcf.gz"
        )
    if n_parts > 0:
        vcf_pipeline_utils.combine_vcf(sp, n_parts, input_prefix, output_fn)
    else:
        output_fn = input_prefix + ".vcf.gz"

    if not output_suffix:
        reheader_fn = pjoin(output_dir, input_prefix_basename + ".rhdr.vcf.gz")
    else:
        reheader_fn = pjoin(
            output_dir, input_prefix_basename + f".{output_suffix}.rhdr.vcf.gz"
        )

    if header is not None:
        vcf_pipeline_utils.reheader_vcf(sp, output_fn, header, reheader_fn)
    else:
        shutil.copy(output_fn, reheader_fn)
        shutil.copy(".".join((output_fn, "tbi")), ".".join((reheader_fn, "tbi")))

    if not output_suffix:
        select_intervals_fn = pjoin(
            output_dir, input_prefix_basename + ".intsct.vcf.gz"
        )
    else:
        select_intervals_fn = pjoin(
            output_dir, input_prefix_basename + f".{output_suffix}.intsct.vcf.gz"
        )

    if not cmp_intervals.is_none():
        vcf_pipeline_utils.intersect_with_intervals(sp,
            reheader_fn, cmp_intervals.as_interval_list_file(), select_intervals_fn
        )
    else:
        shutil.copy(reheader_fn, select_intervals_fn)
        vcf_pipeline_utils.index_vcf(sp, select_intervals_fn)

    if output_file_name is None:
        output_prefix = select_intervals_fn[
            : select_intervals_fn.index(".intsct.vcf.gz")
        ]
    else:
        output_prefix = splitext(output_file_name)[0]

    if concordance_tool == "VCFEVAL":
        vcf_pipeline_utils.run_vcfeval_concordance(
            sp,
            select_intervals_fn,
            truth_file,
            output_prefix,
            ref_genome,
            cmp_intervals.as_interval_list_file(),
            call_sample,
            truth_sample,
            ignore_filter,
        )
        output_prefix = f"{output_prefix}.vcfeval_concordance"
    else:
        vcf_pipeline_utils.run_genotype_concordance(
            sp,
            select_intervals_fn,
            truth_file,
            output_prefix,
            cmp_intervals.as_interval_list_file(),
            call_sample,
            truth_sample,
            ignore_filter,
        )
        output_prefix = f"{output_prefix}.genotype_concordance"

    vcf_pipeline_utils.annotate_tandem_repeats(sp, output_prefix + ".vcf.gz", ref_genome)
    output_prefix = f"{output_prefix}.annotated"

    vcf_pipeline_utils.filter_bad_areas(
        sp,
        select_intervals_fn,
        highconf_intervals.as_bed_file(),
        runs_intervals.as_bed_file(),
    )
    vcf_pipeline_utils.filter_bad_areas(
        sp,
        output_prefix + ".vcf.gz",
        highconf_intervals.as_bed_file(),
        runs_intervals.as_bed_file(),
    )

    if not runs_intervals.is_none():
        return (
            select_intervals_fn.replace("vcf.gz", "runs.vcf.gz"),
            output_prefix + ".runs.vcf.gz",
        )
    else:
        return (
            select_intervals_fn.replace("vcf.gz", "highconf.vcf.gz"),
            output_prefix + ".highconf.vcf.gz",
        )
