import pathmagic
import python.error_model as error_model
import argparse
import os
from os.path import dirname

ap = argparse.ArgumentParser(prog="add_ml_tags_bam.py", description="Add probability tags to uBAM")
ap.add_argument("--probability_tensor", help='Probability tensor (npy/bin)', type=str)
ap.add_argument("--probability_tensor_sequence",
                help='Probability matrix for sequence generation', type=str, required=False)
ap.add_argument("--regressed_key", help='Regressed key (npy/bin)', required=False, type=str)
ap.add_argument("--input_ubam", help='Input uBAM file', required=True, type=str)
ap.add_argument("--output_ubam", help='Output uBAM file', required=True, type=str)
ap.add_argument("--flow_order", help="Flow cycle", default="TACG", type=str)
ap.add_argument("--n_flows", help='Number of flows (required if probability tensor or regressed key are bin)',
                type=int, required=False, default=None)
ap.add_argument("--n_classes",
                help='Number of probability classes (required if probability tensor or regressed key are bin)',
                type=int, required=False, default=None)
ap.add_argument("--probability_threshold", help="Minimal probability to report",
                required=False, default=0.003, type=float)

args = ap.parse_args()


assert (args.probability_tensor.endswith("npy") and (args.regressed_key and args.regressed_key.endswith("npy")))\
    or (args.n_flows is not None and args.n_classes is not None),\
    "If binary matrices are given as input - number of flows and classes should be given"

assert (not args.regressed_key or not args.probability_tensor_sequence), \
    "regressed key and probability_tensor_sequence should not be given together"
if args.probability_tensor_sequence:
    args.regressed_key = error_model.save_tmp_kr_matrix(args.probability_tensor_sequence, 
      args.n_classes, args.n_flows, dirname(args.output_ubam))

matrix_file_name = ".".join((args.probability_tensor, "output.matrix.txt"))

error_model.write_matrix_tags(tensor_name=args.probability_tensor,
                              key_name=args.regressed_key,
                              output_file=matrix_file_name,
                              n_flows=args.n_flows,
                              n_classes=args.n_classes,
                              probability_threshold=args.probability_threshold)


if args.probability_tensor_sequence:
    seq_file_name = '.'.join((args.probability_tensor, "output.seq.txt"))
    error_model.write_sequences(args.probability_tensor_sequence, seq_file_name,
                                args.n_flows, args.n_classes, args.flow_order)
elif args.regressed_key is None:
    seq_file_name = '.'.join((args.probability_tensor, "output.seq.txt"))
    error_model.write_sequences(args.probability_tensor, seq_file_name, args.n_flows, args.n_classes, args.flow_order)


if args.probability_tensor_sequence:
    os.unlink(args.regressed_key)


error_model.add_matrix_to_bam(args.input_ubam, matrix_file_name, args.output_ubam, seq_file_name)
os.unlink(matrix_file_name)

if args.regressed_key is None or args.probability_tensor_sequence:
    os.unlink(seq_file_name)
