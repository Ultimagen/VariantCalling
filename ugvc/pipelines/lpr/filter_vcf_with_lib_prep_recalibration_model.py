import os

from simppl.cli import get_parser, get_simple_pipeline


def init_parser():
    parser = get_parser("filter_vcf_with_lib_prep_recalibration_model", run.__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_suffix", required=True)
    parser.add_argument("--ref_fasta", required=True)
    parser.add_argument("--lib_prep_model_file", required=True)
    parser.add_argument("--notebooks_dir", required=True)
    parser.add_argument(
        "--clustered_calls_parquet", help="filtered variant calling clustered by qual/af", required=True
    )
    parser.add_argument("--truth_vcf", help='set of "true" calls vcf', required=True)
    parser.add_argument("--featuremap_on_calls_parquet", help="featuremap vcf intersected on calls", required=True)
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

    clustered_calls_parquet = args.clustered_calls_parquet
    truth_vcf = args.truth_vcf
    featuremap_on_calls_parquet = args.featuremap_on_calls_parquet

    tp_positions = f"{out_dir}/true_positions.csv"
    sp.print_and_run(
        f"bcftools view {truth_vcf} -H -f PASS -i 'QUAL>{args.min_truth_qual}' "
        "| awk -v OFS=',' '{print $1,$2,$4,$5}' > " + tp_positions
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
        f"-p vcf_input {clustered_calls_parquet} "
        f"-p top_n_read_scores_input "
        f"{out_dir}/recalibration_outputs/{model_name}.scored_alleles.parquet "
        f"-p tp_positions_file {tp_positions} "
    )

    sp.print_and_run(
        f"jupyter nbconvert --to html {out_dir}/score_vcf_with_lib_prep_calibration_data_{out_suffix}.nbconvert.ipynb"
        f" --no-input --output score_vcf_with_lib_prep_calibration_data_{out_suffix}.html"
    )
