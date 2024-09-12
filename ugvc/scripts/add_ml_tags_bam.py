import argparse
import os
from os.path import dirname

from python import error_model

from ugbio_core.consts import DEFAULT_FLOW_ORDER

ap = argparse.ArgumentParser(prog="add_ml_tags_bam.py", description="Add probability tags to uBAM")
ap.add_argument("--probability_tensor", help="Probability tensor (npy/bin)", type=str)
ap.add_argument(
    "--probability_tensor_sequence",
    help="Probability matrix for sequence generation",
    type=str,
    required=False,
)
ap.add_argument("--regressed_key", help="Regressed key (npy/bin)", required=False, type=str)
ap.add_argument("--input_ubam", help="Input uBAM file", required=True, type=str)
ap.add_argument("--output_ubam", help="Output uBAM file", required=True, type=str)
ap.add_argument("--flow_order", help="Flow cycle", default=DEFAULT_FLOW_ORDER, type=str)
ap.add_argument(
    "--n_flows",
    help="Number of flows (required if probability tensor or regressed key are bin)",
    type=int,
    required=False,
    default=None,
)
ap.add_argument(
    "--n_classes",
    help="Number of probability classes (required if probability tensor or regressed key are bin)",
    type=int,
    required=False,
    default=None,
)
ap.add_argument(
    "--probability_threshold",
    help="Minimal probability to report",
    required=False,
    default=0.003,
    type=float,
)
ap.add_argument(
    "--probability_scaling_factor",
    help="Probability scaling factor",
    default=10,
    type=float,
)

args = ap.parse_args()


assert (args.probability_tensor.endswith("npy") and (args.regressed_key and args.regressed_key.endswith("npy"))) or (
    args.n_flows is not None and args.n_classes is not None
), "If binary matrices are given as input - number of flows and classes should be given"

assert (
    not args.regressed_key or not args.probability_tensor_sequence
), "regressed key and probability_tensor_sequence should not be given together"
if args.probability_tensor_sequence:
    args.regressed_key = error_model.save_tmp_kr_matrix(
        args.probability_tensor_sequence,
        args.n_classes,
        args.n_flows,
        dirname(args.output_ubam),
    )

MATRIX_FILE_NAME = ".".join((args.probability_tensor, "output.matrix.txt"))

print("Writing matrix tags")
empty, complete = error_model.write_matrix_tags(
    tensor_name=args.probability_tensor,
    key_name=args.regressed_key,
    output_file=MATRIX_FILE_NAME,
    n_flows=args.n_flows,
    n_classes=args.n_classes,
    probability_threshold=args.probability_threshold,
    probability_sf=args.probability_scaling_factor,
)
print("Wrote", complete, "complete tags and", empty, "empty lines")
if args.probability_tensor_sequence:
    print("Writing sequences")

    SEQ_FILE_NAME = ".".join((args.probability_tensor, "output.seq.txt"))
    n_written = error_model.write_sequences(
        args.probability_tensor_sequence,
        SEQ_FILE_NAME,
        args.n_flows,
        args.n_classes,
        args.flow_order,
    )
    print("Wrote", n_written, "tags")

elif args.regressed_key is None:
    print("Writing sequences")
    SEQ_FILE_NAME = ".".join((args.probability_tensor, "output.seq.txt"))
    n_written = error_model.write_sequences(
        args.probability_tensor,
        SEQ_FILE_NAME,
        args.n_flows,
        args.n_classes,
        args.flow_order,
    )
    print("Wrote", n_written, "tags")


if args.probability_tensor_sequence:
    os.unlink(args.regressed_key)


error_model.add_matrix_to_bam(args.input_ubam, MATRIX_FILE_NAME, args.output_ubam, SEQ_FILE_NAME)
os.unlink(MATRIX_FILE_NAME)

if args.regressed_key is None or args.probability_tensor_sequence:
    os.unlink(SEQ_FILE_NAME)
