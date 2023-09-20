#!/env/python
# Copyright 2023 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Converts featuremap VCF-like file to dataframe
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse

# from ugvc.mrd.mrd_utils import default_featuremap_info_fields, featuremap_to_dataframe
from ugvc.mrd.mrd_utils import featuremap_to_dataframe


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="featuremap_to_dataframe", description=run.__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input featuremap file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""Path to which output dataframe will be written, if None a file with the same name as input and
".parquet" extension will be created""",
    )
    parser.add_argument(
        "-if",
        "--info_fields_override",
        type=str,
        nargs="+",
        required=False,
        default="all",
        help="""List of input info fields to include in dataframe, by default 'all''
        If 'all' then all the info fields are read to columns
        If None then no info fields are read to columns""",
    )

    parser.add_argument(
        "--input-format-fields", type=str, nargs="+", default=None, help="Fields to extract from the vcf FORMAT fields"
    )
    return parser.parse_args(argv[1:])


# def __parse_dict_from_arg(arg: list[str]) -> dict:
#     """
#     parse fields command-line arguments where each argument is passed as key=val,
#     """
#     if arg is None:
#         return {}
#     d = {}
#     # wasn't splitted correctly, such as when calling from simppl print_and_run_clt
#     if len(arg) == 1 and " " in arg[0]:
#         arg = arg[0].split(" ")
#     for f in arg:
#         if f == "":
#             continue
#         key, val = f.split("=")
#         d[key] = val
#     return d


def run(argv: list[str]):
    """Convert featuremap to pandas dataframe"""
    args_in = __parse_args(argv)

    # if args_in.info_fields_override is None:
    #     info_fields = default_featuremap_info_fields
    # else:
    #     info_fields = __parse_dict_from_arg(args_in.info_fields_override)
    # info_fields.update(__parse_dict_from_arg(args_in.extra_fields))
    # input_format_fields = __parse_dict_from_arg(args_in.input_format_fields)

    featuremap_to_dataframe(
        featuremap_vcf=args_in.input,
        output_file=args_in.output,
        # input_format_fields=input_format_fields,
        # input_info_fields=info_fields,
    )
