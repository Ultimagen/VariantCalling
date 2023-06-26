import simppl
import argparse
import os

def __get_parser(argv: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training_set_consistency_check", description=run.__doc__)
    parser.add_argument(
        "--target_crams",
        type=str,
        required=True,
        help="crams to call variants on, comprising a single (hopefully consistent) training sub-set",
    )
    parser.add_argument(
        "--ground_truth_vcf",
        type=str,
        required=True,
        help="vcf-file with ground truth variants"
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="fasta reference genome",
    )
    parser.add_argument(
        "--normal_crams",
        type=str,
        help="crams of normal samples, used for training somatic models",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="output directory"
    )
    return parser


def vcf_to_bed(vcf: str, bed: str, sp: SimplePipeline) -> None:
    sp.print_and_run(f"bcftools view -H --type snps {vcf} | awk -v OFS='\t' '{print $1,$2,$2,$5}' > {bed}")

def call_variants(cram: str, ref: str, vcf: str, regions: str, sp: SimplePipeline) -> None:
    sp.print_and_run(f"bcftools mpileup {cram} -f {ref} -T {regions} | bcftools call -mv -Oz -o {vcf}")

def read_count(file: str) -> int:
    with open(f'{args.out_dir}/counts') as f:
            return int(f.read().strip())

def calc_hit_fraction(cram: str, ref: str, ground_truth_bed: str, ground_truth_count: int) -> float:
    cram_base_name = os.path.basename(cram)
    vcf = f'{args.out_dir}/{cram_base_name}.vcf.gz'
    bed = f'{args.out_dir}/{cram_base_name}.bed'
    counts = f'{args.out_dir}/{cram_base_name}.counts'
    call_variants(cram, ref, vcf, ground_truth_bed, sp)
    vcf_to_bed(vc, bed, sp)
    sp.print_and_run(f'wc -l {bed} > {counts}')
    hit_count = read_count(counts)
    hit_fraction = hit_count / ground_truth_count
    return hit_fraction


def run(argv):
    """Training set consistency check pipeline."""
    parser = get_parser()
    SimplePipeline.add_parse_args(parser)
    args = parser.parse_args(argv[1:])

    target_crams = args.target_crams.split(',')
    if args.normal_crams:
        normal_crams = args.normal_crams.split(',')
    else:
        normal_crams = None
    ground_truth_vcf = args.ground_truth_vcf
    ref = args.ref

    sp = SimplePipeline(args.fc, args.lc, debug=args.d) 
    errors = []

    ground_truth_bed = f'{args.out_dir}/ground_truth.bed'
    sp.print_and_run(f"{vcf_to_bed_command(ground_truth_vcf)} > {ground_truth_bed}")
    sp.print_and_run(f"wc -l {ground_truth_bed} > {args.out_dir}/ground_truth_count}")
    ground_truth_count = read_count(f'{args.out_dir}/ground_truth_count')

    for target_cram in target_crams:
        hit_fraction = calc_hit_fraction(target_cram, ref, ground_truth_bed, ground_truth_count)
        if hit_fraction < 0.95:
            if normal_crams is None:    
                errors.append((target_cram, f'target sample does not match ground truth hit_fraction={hit_fraction}}'))
            elif hit_fraction > 0.05:
                errors.append((target_cram, f'target sample does not match ground truth, and is also not complementary to it, hit_fraction={hit_fraction}'))
            else:
                suspected_normal_in_tumor_crams.append(target_cram)
    
    for normal_cram in normal_crams:
        hit_fraction = calc_hit_fraction(target_cram, ref, ground_truth_bed, ground_truth_count)
        if hit_fraction > 0.05:
            errors.append((normal_cram, f'normal sample is not complementary to ground truth, hit_fraction={hit_fraction}'))
        
    for susected_normal_in_tumor_cram, ground_truth_hit_fraction in suspected_normal_in_tumor_crams:
        found_matched_normal = False
        max_hit_fraction = 0
        for normal_cram in normal_crams:
            normal_base_name = os.path.basename(normal_cram)
            normal_cram_bed = f'{args.out_dir}/{normal_base_name}.bed'
            hit_fraction = calc_hit_fraction(target_cram, ref, normal_cram_bed, ground_truth_count)
            if hit_fraction > 0.95:
                found_matched_normal = True
            else:
                max_hit_fraction = max(max_hit_fraction, hit_fraction)
        if max_hit_fraction < 0.95:
            errors.append((suspected_notmal_in_tumor_cram, f'suspected normal_in_tumor sample does not match any normal sample max_hit_fraction={max_hit_fraction}'))
    
    if len(errors) > 0:
        raise RuntimeError('\n'.join(errors))
