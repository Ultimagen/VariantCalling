import argparse
import convert_h5_to_json_utils
import json
import os
import pandas as pd
import sys
import tables

ap = argparse.ArgumentParser(
    prog="convert_h5_to_json.py", description="Convert h5 file to json file (using pandas table serialization) optionally ready for MongoDB")
ap.add_argument("--input_h5", help='Metrics input h5 file name',
            required=True, type=str)
ap.add_argument("--output_json", help='Metrics output json file name',
            required=True, type=str)
ap.add_argument("--root_element", help="Root element of the output JSON",
            required=True, type=str)
ap.add_argument("--ignored_h5_key_substring", help="The h5 key substring to ignore during conversion",
            required=False, type=str)
ap.add_argument("--mongodb_preprocessing", help="Prepare for MongoDB (for example: remove '$' from keys)",
            required=False, type=bool, default=True)

args = ap.parse_args()
input_h5_filename = args.input_h5
output_json_filename = args.output_json
mongodb_preprocessing = args.mongodb_preprocessing
ignored_h5_key_substring = args.ignored_h5_key_substring
root_element = args.root_element

json_string = convert_h5_to_json_utils.convert_h5_to_json(input_h5_filename, root_element, mongodb_preprocessing, ignored_h5_key_substring)
f = open(output_json_filename, "w")
f.write(json_string)
f.close()
