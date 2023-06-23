import os

from simppl.cli import get_parser, get_simple_pipeline

from ugvc.pipelines.mrd import featuremap_to_dataframe


def init_parser():
    parser = get_parser("filter_vcf_with_lib_prep_recalibration_model", run.__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_suffix", required=True)
    parser.add_argument("--ref_fasta", required=True)
    parser.add_argument("--lib_prep_model_file", required=True)
    parser.add_argument("--notebooks_dir", required=True)
    parser.add_argument("--filtered_calls_vcf", help="filtered variant calling input vcf", required=True)
    parser.add_argument("--truth_vcf", help='set of "true" calls vcf', required=True)
    parser.add_argument("--featuremap_on_calls_vcf", help="featuremap vcf intersected on calls", required=True)
    parser.add_argument(
        "--min_truth_qual", type=int, default=10, help="min qual to consider truth vcf variant as truth"
    )
    return parser


def run(argv):
    """
    Filter vcf file using lib-prep recalibration model training pipeline
    """
    parser = init_parser()
    sp = get_simple_pipeline(parser, argv=argv, name=__name__)
    args = parser.parse_args(argv[1:])
    print(args)

    out_dir = args.out_dir
    out_suffix = args.output_suffix
    os.makedirs(out_dir, exist_ok=True)

    filtered_calls_vcf = args.filtered_calls_vcf
    truth_vcf = args.truth_vcf
    featuremap_on_calls_vcf = args.featuremap_on_calls_vcf

    calls_parquet = f"{out_dir}/calls_vcf.parquet"
    featuremap_on_calls_parquet = f"{out_dir}/featuremap_on_calls.parquet"

    sp.print_and_run_clt(
        featuremap_to_dataframe.run,
        [],
        {
            "-i": filtered_calls_vcf,
            "-r": args.ref_fasta,
            "-f": "TGCA",
            "--info_fields_override": "",
            "--format-fields": "DP=int AD=str SB=str VAF=str PL=str BG_AD=str",
            "-o": calls_parquet,
        },
        flags=["--report-bases-in-reference-direction"],
    )

    sp.print_and_run_clt(
        featuremap_to_dataframe.run,
        [],
        {"-i": featuremap_on_calls_vcf, "-r": args.ref_fasta, "-f": "TGCA", "-o": featuremap_on_calls_parquet},
    )

    tp_positions = f"{out_dir}/true_positions.csv"
    sp.print_and_run(
        f"bcftools view {truth_vcf} -H -f PASS -i 'QUAL>{args.truth_qual_thresh}' "
        "| awk '{print $2}' > " + tp_positions
    )

    base_dir = args.notebooks_dir
    model_name = os.path.basename(args.lib_prep_model_file)
    sp.print_and_run(
        f"papermill {base_dir}/lib_prep_recalibration_N_top_reads_per_allele.ipynb "
        f"{out_dir}/lib_prep_recalibration_N_top_reads_per_allele.nbconvert.ipynb "
        f"-p model {args.lib_prep_model_file} "
        f"-p featuremap {featuremap_on_calls_parquet} "
        f"-p out_dir {out_dir}/recalibration_outputs"
    )

    sp.print_and_run(
        f"papermill {base_dir}/score_vcf_with_lib_prep_calibration_data.ipynb "
        f"{out_dir}/score_vcf_with_lib_prep_calibration_data_{out_suffix}.nbconvert.ipynb "
        f"-p work_dir {out_dir} "
        f"-p vcf_input {calls_parquet} "
        f"-p top_n_read_scores_input "
        f"{out_dir}/recalibration_outputs/{model_name}.scored_alleles.parquet "
        f"-p tp_positions_file {tp_positions} "
    )

    sp.print_and_run(
        f"jupyter nbconvert --to html {out_dir}/score_vcf_with_lib_prep_calibration_data_{out_suffix}.nbconvert.ipynb"
        f" --no-input --output score_vcf_with_lib_prep_calibration_data_{out_suffix}.html"
    )
