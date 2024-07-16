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
#    Run single read SNV quality recalibration training
# CHANGELOG in reverse chronological order
from __future__ import annotations

import argparse
import json

from simppl.simple_pipeline import SimplePipeline

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.mrd.ppmSeq_utils import supported_adapter_versions
from ugvc.mrd.srsnv_plotting_utils import srsnv_report
from ugvc.mrd.srsnv_training_utils import SRSNVTrain


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="srsnv_training.py", description=run.__doc__)
    parser.add_argument(
        "--hom_snv_featuremap",
        type=str,
        required=True,
        help="""input featuremap with homozygote SNVs (True Positive)""",
    )
    parser.add_argument(
        "--single_substitution_featuremap",
        type=str,
        required=True,
        help="""single substitution featuremap (False Positive)""",
    )
    parser.add_argument(
        "--hom_snv_regions",
        type=str,
        required=False,
        help="""Path to bed file containint regions for hom snv (TP) featuremap""",
    )
    parser.add_argument(
        "--single_sub_regions",
        type=str,
        required=True,
        help="""Path to bed file containint regions for single substitution featuremap (FP)""",
    )
    parser.add_argument(
        "--cram_stats_file",
        type=str,
        required=True,
        help="""Path to cram stats file (for LoD estimation)""",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        required=False,
        help="""Path to json file with input parameters for the classification model""",
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        help="""Size of the train set for the classification model""",
    )
    parser.add_argument(
        "--test_set_size",
        type=int,
        default=None,
        help="""Size of the test set for the classification model""",
    )
    parser.add_argument(
        "--num_CV_folds",
        type=int,
        default=None,
        help="""Number of cross-validation folds to use. Default=1 (no CV)""",
    )
    parser.add_argument(
        "--split_folds_randomly",
        action="store_true",
        help="""by default the training data is split into folds by chromosomes,
        if the flag is provided it is split randomly.""",
    )
    parser.add_argument(
        "--numerical_features",
        type=str,
        nargs="+",
        help="""comma separated list of numerical features for ML classifier """,
    )
    parser.add_argument(
        "--categorical_features",
        type=str,
        nargs="+",
        help="""comma separated list of categorical features for ML classifier.
        Each item is a dictionary, e.g.: '{"ref": ["T", "G", "C", "A"]}' """,
    )
    parser.add_argument(
        "--dataset_params_json_path",
        type=str,
        required=False,
        default=None,
        help="""Path to JSON file that contains dataset parameters, i.e.
        train_set_size, test_set_size, numerical_features, categorical_features,
        balanced_sampling_info_fields, pre_filter""",
    )
    parser.add_argument(
        "--balanced_sampling_info_fields",
        type=str,
        nargs="+",
        default=None,
        help="comma separated list of categorical features to be used for balanced sampling of the TP training set"
        " to eliminate prior distribution bias (e.g. 'trinuc_context_with_alt,is_forward')",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Path to which output files will be written to""",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="",
        required=False,
        help="""basename of output files""",
    )
    parser.add_argument(
        "--save_model_jsons",
        action="store_true",
        help="""Save model(s) in json files. By default models are saved only
        as a single joblib file.""",
    )
    parser.add_argument(
        "-r",
        "--reference_fasta",
        type=str,
        help="""reference fasta, only required for motif annotation""",
    )
    parser.add_argument(
        "--reference_dict",
        type=str,
        help="""reference dict, required to know chromosome sizes""",
    )
    parser.add_argument(
        "--flow_order",
        type=str,
        required=False,
        default=DEFAULT_FLOW_ORDER,
        help="""flow order - required for cycle skip annotation """,
    )
    parser.add_argument(
        "--lod_filters",
        type=str,
        required=False,
        default=None,
        help="""json file with a dict of format 'filter name':'query' for LoD simulation """,
    )
    parser.add_argument(
        "--ppmSeq_adapter_version",
        choices=supported_adapter_versions,
        required=False,
        help="ppmSeq adapter version",
    )
    parser.add_argument(
        "--pre_filter",
        type=str,
        required=False,
        default=None,
        help="""bcftools include filter to apply as part of a "bcftools view <vcf> -i 'pre_filter' command""",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default=None,
        help="""random seed for reproducibility""",
    )
    return parser.parse_args(argv[1:])


# pylint:disable=missing-raises-doc
def read_dataset_params(args):
    """Read the dataset params from json file. Any values provided in the command line
    overrides the json file.
    Arguments:
        - args: the command line arguments.
    Returns:
        - dataset_params [dict]: dictionary with dataset params.

    Raises:
        - ValueError, if split_folds_by has value other than 'random' or 'chrom'
    """
    dataset_params = {
        "train_set_size": args.train_set_size,
        "test_set_size": args.test_set_size,
        "numerical_features": args.numerical_features,
        "categorical_features": args.categorical_features,
        "balanced_sampling_info_fields": args.balanced_sampling_info_fields,
        "pre_filter": args.pre_filter,
        "ppmSeq_adapter_version": args.ppmSeq_adapter_version,
        "random_seed": args.random_seed,
        "num_CV_folds": args.num_CV_folds,
    }
    if args.categorical_features:
        # convert string to dictionary:
        dataset_params["categorical_features"] = {
            k: v for feat in args.categorical_features for k, v in json.loads(feat).items()
        }
    if args.dataset_params_json_path is not None:
        # log that loading params from json
        with open(args.dataset_params_json_path, encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = {}

    dataset_params = {p: v or params.get(p, None) for p, v in dataset_params.items()}
    # Add boolean features to categorical
    if params.get("boolean_features", None) is not None:
        for feat in params["boolean_features"]:
            dataset_params["categorical_features"][feat] = [False, True]
    # default value of num_CV_folds
    dataset_params["num_CV_folds"] = dataset_params["num_CV_folds"] or 1
    # default value of random_seed
    dataset_params["random_seed"] = dataset_params["random_seed"] or 42
    # check split_folds_by_chrom
    dataset_params["split_folds_by_chrom"] = True
    if params.get("split_folds_by", None) is not None:
        if params["split_folds_by"] == "chrom":
            dataset_params["split_folds_by_chrom"] = True
        elif params["split_folds_by"] == "random":
            dataset_params["split_folds_by_chrom"] = False
        else:
            raise ValueError(
                f"split_folds_by can only have values 'chrome' and 'random'. Got {params['split_folds_by']}"
            )
    if args.split_folds_randomly:  # override json file
        dataset_params["split_folds_by_chrom"] = False

    return dataset_params


def run(argv: list[str]):
    """Train a model for single read SNV quality recalibration"""
    args = parse_args(argv)
    simple_pipeline_args = (0, 10000, False)
    sp = SimplePipeline(
        simple_pipeline_args[0],
        simple_pipeline_args[1],
        debug=simple_pipeline_args[2],
        print_timing=True,
    )

    # TODO add the option to read from a json file         model_parameters: dict | str = None,
    # TODO add to args         classifier_class=xgb.XGBClassifier,

    dataset_params = read_dataset_params(args)

    s = SRSNVTrain(
        tp_featuremap=args.hom_snv_featuremap,
        fp_featuremap=args.single_substitution_featuremap,
        tp_regions_bed_file=args.hom_snv_regions,
        fp_regions_bed_file=args.single_sub_regions,
        numerical_features=dataset_params["numerical_features"],
        categorical_features=dataset_params["categorical_features"],
        balanced_sampling_info_fields=(
            dataset_params["balanced_sampling_info_fields"] if dataset_params["balanced_sampling_info_fields"] else None
        ),
        sorter_json_stats_file=args.cram_stats_file,
        train_set_size=dataset_params["train_set_size"],
        test_set_size=dataset_params["test_set_size"],
        k_folds=dataset_params["num_CV_folds"],
        split_folds_by_chrom=dataset_params["split_folds_by_chrom"],
        model_params=args.model_params,
        reference_dict=args.reference_dict,
        out_path=args.output,
        out_basename=args.basename,
        lod_filters=args.lod_filters,
        save_model_jsons=args.save_model_jsons,
        ppmSeq_adapter_version=dataset_params["ppmSeq_adapter_version"],
        pre_filter=dataset_params["pre_filter"],
        random_seed=dataset_params["random_seed"],
        simple_pipeline=sp,
    ).process()

    # TODO: merge the two reports so train and test set results are presented together
    srsnv_report(
        out_path=args.output,
        out_basename=args.basename,
        report_name="test",
        model_file=s.model_joblib_save_path,
        params_file=s.params_save_path,
        simple_pipeline=None,
    )

    srsnv_report(
        out_path=args.output,
        out_basename=args.basename,
        report_name="train",
        model_file=s.model_joblib_save_path,
        params_file=s.params_save_path,
        simple_pipeline=None,
    )
