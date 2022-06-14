#!/env/python
import argparse

from ugvc.utils import metrics_utils

ap = argparse.ArgumentParser(
    prog="convert_h5_to_json.py",
    description="Convert h5 file to json file (using pandas table serialization) optionally ready for MongoDB",
)
ap.add_argument("--input_h5", help="Metrics input h5 file name", required=True, type=str)
ap.add_argument("--output_json", help="Metrics output json file name", required=True, type=str)
ap.add_argument("--root_element", help="Root element of the output JSON", required=True, type=str)
ap.add_argument(
    "--ignored_h5_key_substring",
    help="The h5 key substring to ignore during conversion",
    required=False,
    type=str,
)

args = ap.parse_args()
input_h5_filename = args.input_h5
output_json_filename = args.output_json
ignored_h5_key_substring = args.ignored_h5_key_substring
root_element = args.root_element

json_string = metrics_utils.convert_h5_to_json(input_h5_filename, root_element, ignored_h5_key_substring)
with open(output_json_filename, "w", encoding="utf-8") as f:
    f.write(json_string)
    f.close()
