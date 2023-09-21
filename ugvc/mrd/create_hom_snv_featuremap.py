import os

from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugvc.mrd.featuremap_utils import FeatureMapFields
from ugvc.utils.exec_utils import print_and_execute
from ugvc.utils.metrics_utils import read_effective_coverage_from_sorter_json


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
