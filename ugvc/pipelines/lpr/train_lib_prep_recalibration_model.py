import os

from simppl.cli import get_parser, get_simple_pipeline

from ugvc.pipelines.mrd import featuremap_to_dataframe
from ugvc.utils.cloud_sync import cloud_sync


def init_parser():
    parser = get_parser("train_lib_prep_recalibration_model", run.__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ref_fasta", required=True)
    parser.add_argument("--featuremap_vcf", help="full featuremap vcf file")
    parser.add_argument("--notebooks_dir", required=True)
    parser.add_argument("--calls_vcf", help="variant calling vcf file")
    parser.add_argument("--tp_min_af", type=float, help="min allele-frequency to consider a variant tp", default=0.9)
    parser.add_argument("--fp_max_af", type=float, help="max allele-count to consider a variant fp", default=0.04)
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--balance_motifs", default=False, action="store_true")
    parser.add_argument("--balance_tp_fp", default=False, action="store_true")

    return parser


def run(argv):
    """
    Lib-prep recalibration model training pipeline
    """
    parser = init_parser()
    sp = get_simple_pipeline(parser, argv=argv, name=__name__)
    args = parser.parse_args(argv[1:])
    print(args)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    featuremap_vcf, unfiltered_calls = localize_inputs(args)

    tp_args = [f"{out_dir}/estimated_tps.{suffix}" for suffix in ("bed", "vcf.gz", "parquet")]
    fp_args = [f"{out_dir}/estimated_fps.{suffix}" for suffix in ("bed", "vcf.gz", "parquet")]
    tp_args.extend([args.tp_min_af, 1, args.tp_min_qual])
    fp_args.extend([0, args.fp_max_af, args.fp_min_qual])
    afs_bed = f"{out_dir}/afs.bed"

    if args.calls_vcf:
        calls_vcf = f"{out_dir}/calls.snps.pass.vcf.gz"
        featuremap_on_calls = f"{out_dir}/featuremap_on_calls.vcf.gz"
        sp.print_and_run(f"bcftools view --type snps -f PASS -m2 -M2 {unfiltered_calls} -Oz -o {calls_vcf}")
        sp.print_and_run(
            f"bedtools intersect -a {featuremap_vcf} -b {calls_vcf} -header | "
            f"bcftools view -Oz -o {featuremap_on_calls}"
        )
        featuremap_vcf = featuremap_on_calls
        sp.print_and_run(f"bcftools query -f '%CHROM\t%POS0\t%POS\t[%VAF]\\n' {calls_vcf} | grep -v , > {afs_bed}")
    else:
        sp.print_and_run(
            "bcftools query -f '%CHROM\t%POS0\t%POS\t%X_READ_COUNT' "
            " | bedtools groupby -c 3 -full -o count | awk -v OFS='\t' '{print $1,$2,$3,$5/$4}' " + f" > {afs_bed}"
        )

    for bed_file, vcf_file, parquet_file, min_af, max_af in (tp_args, fp_args):
        sp.print_and_run(f"awk '$4 >= {min_af} && $4 <= {max_af}' {afs_bed} > {bed_file}")
        sp.print_and_run(
            f"bedtools intersect -a {featuremap_vcf} -b {bed_file} -u -header | " f"bcftools view -Oz -o {vcf_file}"
        )
        sp.print_and_run_clt(
            featuremap_to_dataframe.run, [], {"-i": vcf_file, "-r": args.ref_fasta, "-f": "TGCA", "-o": parquet_file}
        )

    base_dir = args.notebooks_dir
    balance_motifs = "True" if args.balance_motifs else "False"
    balance_tp_fp = "True" if args.balance_tp_fp else "False"
    output_suffix_arg = args.output_suffix if args.output_suffix != "" else '""'
    sp.print_and_run(
        f"papermill {base_dir}/train_lib_prep_xgboost.ipynb "
        f"{out_dir}/train_lib_prep_xgboost{args.output_suffix}.nbconvert.ipynb "
        f"-p work_dir {out_dir} "
        f"-p suffix {output_suffix_arg} "
        f"-p balance_motifs {balance_motifs} "
        f"-p balance_tp_fp {balance_tp_fp} "
    )

    sp.print_and_run(
        "jupyter nbconvert --to html "
        f"{out_dir}/train_lib_prep_xgboost{args.output_suffix}.nbconvert.ipynb "
        f"--no-input --output train_lib_prep_xgboost_report{args.output_suffix}.html"
    )


def localize_inputs(args):
    featuremap_vcf = args.featuremap_vcf
    calls_vcf = args.calls_vcf
    out_dir = args.out_dir
    if args.calls_vcf.startswith("gs://"):
        calls_vcf = cloud_sync(args.calls_vcf, out_dir)
    if featuremap_vcf.startswith("gs://"):
        featuremap_vcf = cloud_sync(featuremap_vcf, out_dir)
    return featuremap_vcf, calls_vcf
