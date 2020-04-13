import pathmagic
import python.error_model as error_model
import argparse
import os

ap = argparse.ArgumentParser(prog="add_ml_tags_bam.py", description="Add probability tags to uBAM")
grp = ap.add_mutually_exclusive_group(reuired=True)
grp.add_argument("--probability_tensor", help='Probability tensor (npy/bin)', type=str)
grp.add_argument("--probability_tensor_fofn", help='List of npy/bin tensors **in order**', type=str)
ap.add_argument("--regressed_key", help='Regressed key (npy/bin)', required=True, type=str)
ap.add_argument("--input_ubam", help='Input uBAM file', required=True, type=str)
ap.add_argument("--output_ubam", help='Output uBAM file', required=True, type=str)
ap.add_argument("--n_flows", help='Number of flows (required if probability tensor or regressed key are bin)', required=False, type=int, default=None)
ap.add_argument("--n_classes", help='Number of probability classes (required if probability tensor or regressed key are bin)', type=int, required=False, default=None)
ap.add_argument("--probability_threshold", help="Minimal probability to report", required=False, default=0.003, type=float)

args = ap.parse_args()

if 'probability_tensor' in ap : 
	probability_tensor_list = [ap.probability_tensor]

else: 
	probability_tensor_list = [ x.strip() for x in open(ap.probability_tensor_fofn) ]

assert (probability_tensor_list[0].endswith("npy") and args.regressed_key.endswith("npy")) or \
	(args.n_flows is not None and args.n_classes is not None), "If binary matrices are given as input - number of flows and classes should be given"

matrix_file_name = ".".join((probability_tensor_list[0], "output.matrix.txt"))
error_model.write_matrix_tags(tensor_name=probability_tensor_list, 
                             key_name=args.regressed_key, 
                             output_file=matrix_file_name,
                             n_flows=args.n_flows, 
                             n_classes=args.n_classes, 
                             probability_threshold=args.probabilit
                             y_threshold)

error_model.add_matrix_to_bam(args.input_ubam, matrix_file_name, args.output_ubam)
os.unlink(matrix_file_name)
