import python.variant_filtering_utils as variant_filtering_utils
import argparse
import pandas as pd
import pickle


ap = argparse.ArgumentParser(prog="train_models_pipeline.py",
                             description="Train filtering models on the concordance file")
grp = ap.add_mutually_exclusive_group(required=True)
grp.add_argument("--input_file", help="Name of the input h5 file", type=str)
grp.add_argument(
    "--input_fofn", help="Input file containing list of the h5 file names to concatenate", type=str)
ap.add_argument("--output_file", help="Output pkl file",
                type=str, required=True)


args = ap.parse_args()
if 'input_file' in args:
  concordance = pd.read_hdf(args.input_file, "concordance")
else:
  concordance = pd.concat([pd.read_hdf(x.strip(), "concordance")
                           for x in open(args.input_fofn)])

concordance.loc[pd.isnull(concordance['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'
concordance_clean = concordance[(~concordance.close_to_hmer_run) & (~concordance.inside_hmer_run)].copy()


# Unfiltered data
model_no_gt, recall_precision_no_gt = variant_filtering_utils.calculate_unfiltered_model(concordance, "classify")
results_dict = {}
results_dict['unfiltered_ignore_gt_incl_hpol_runs'] = model_no_gt
results_dict['unfiltered_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt

model_gt, recall_precision_gt = variant_filtering_utils.calculate_unfiltered_model(concordance, "classify_gt")
results_dict['unfiltered_include_gt_incl_hpol_runs'] = model_gt
results_dict['unfiltered_recall_precision_include_gt_incl_hpol_runs'] = recall_precision_gt

model_no_gt, recall_precision_no_gt = variant_filtering_utils.calculate_unfiltered_model(concordance_clean, "classify")
results_dict['unfiltered_ignore_gt_excl_hpol_runs'] = model_no_gt
results_dict['unfiltered_recall_precision_ignore_gt_excl_hpol_runs'] = recall_precision_no_gt

model_gt, recall_precision_gt = variant_filtering_utils.calculate_unfiltered_model(concordance_clean, "classify_gt")
results_dict['unfiltered_include_gt_excl_hpol_runs'] = model_gt
results_dict['unfiltered_recall_precision_include_gt_excl_hpol_runs'] = recall_precision_gt


# Thresholding model
models_thr_no_gt, models_reg_thr_no_gt, concordance = \
    variant_filtering_utils.train_threshold_models(concordance, classify_column='classify')
recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(concordance, models_thr_no_gt, "classify")
recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance, models_reg_thr_no_gt, "classify")

results_dict['threshold_model_ignore_gt_incl_hpol_runs'] = models_thr_no_gt, models_reg_thr_no_gt
results_dict['threshold_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
results_dict['threshold_model_precision_recall_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt

models_thr_gt, models_reg_thr_gt, concordance = \
    variant_filtering_utils.train_threshold_models(concordance, classify_column='classify_gt')
recall_precision_gt = variant_filtering_utils.test_decision_tree_model(concordance, models_thr_gt, "classify_gt")
recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance, models_reg_thr_gt, "classify_gt")

results_dict['threshold_model_include_gt_incl_hpol_runs'] = models_thr_gt, models_reg_thr_gt
results_dict['threshold_model_recall_precision_include_gt_incl_hpol_runs'] = recall_precision_gt
results_dict['threshold_model_precision_recall_curve_include_gt_incl_hpol_runs'] = recall_precision_curve_gt

models_thr_no_gt, models_reg_thr_no_gt, concordance_clean = \
    variant_filtering_utils.train_threshold_models(concordance_clean, classify_column='classify')
recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
    concordance_clean, models_thr_no_gt, "classify")
recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance_clean, models_reg_thr_no_gt, "classify")

results_dict['threshold_model_ignore_gt_excl_hpol_runs'] = models_thr_no_gt, models_reg_thr_no_gt
results_dict['threshold_model_recall_precision_ignore_gt_excl_hpol_runs'] = recall_precision_no_gt
results_dict['threshold_model_precision_recall_curve_ignore_gt_excl_hpol_runs'] = recall_precision_curve_no_gt

models_thr_gt, models_reg_thr_gt, concordance_clean = \
    variant_filtering_utils.train_threshold_models(concordance_clean, classify_column='classify_gt')
recall_precision_gt = variant_filtering_utils.test_decision_tree_model(concordance_clean, models_thr_gt, "classify_gt")
recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance_clean, models_reg_thr_gt, "classify_gt")

results_dict['threshold_model_include_gt_excl_hpol_runs'] = models_thr_gt, models_reg_thr_gt
results_dict['threshold_model_recall_precision_include_gt_excl_hpol_runs'] = recall_precision_gt
results_dict['threshold_model_precision_recall_curve_include_gt_excl_hpol_runs'] = recall_precision_curve_gt


# Decision tree models
models_dt_no_gt, models_reg_dt_no_gt, concordance = \
    variant_filtering_utils.train_decision_tree_model(concordance,
                                                      classify_column='classify')
recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
    concordance, models_dt_no_gt, "classify")
recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance, models_reg_dt_no_gt, "classify")

results_dict['dt_model_ignore_gt_incl_hpol_runs'] = models_dt_no_gt, models_reg_dt_no_gt
results_dict['dt_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
results_dict['dt_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt


models_dt_gt, models_reg_dt_gt, concordance = \
    variant_filtering_utils.train_decision_tree_model(concordance,
                                                      classify_column='classify_gt')
recall_precision_gt = variant_filtering_utils.test_decision_tree_model(concordance, models_dt_gt, "classify_gt")
recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance, models_reg_dt_gt, "classify_gt")

results_dict['dt_model_include_gt_incl_hpol_runs'] = models_dt_gt, models_reg_dt_gt
results_dict['dt_model_recall_precision_include_gt_incl_hpol_runs'] = recall_precision_gt
results_dict['dt_model_recall_precision_curve_include_gt_incl_hpol_runs'] = recall_precision_curve_gt


models_dt_no_gt, models_reg_dt_no_gt, concordance_clean = \
    variant_filtering_utils.train_decision_tree_model(concordance_clean,
                                                      classify_column='classify')
recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
    concordance_clean, models_dt_no_gt, "classify")
recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance_clean, models_reg_dt_no_gt, "classify")

results_dict['dt_model_ignore_gt_excl_hpol_runs'] = models_dt_no_gt, models_reg_dt_no_gt
results_dict['dt_model_recall_precision_ignore_gt_excl_hpol_runs'] = recall_precision_no_gt
results_dict['dt_model_recall_precision_curve_ignore_gt_excl_hpol_runs'] = recall_precision_curve_no_gt


models_dt_gt, models_reg_dt_gt, concordance_clean = \
    variant_filtering_utils.train_decision_tree_model(concordance_clean,
                                                      classify_column='classify_gt')
recall_precision_gt = variant_filtering_utils.test_decision_tree_model(concordance_clean, models_dt_gt, "classify_gt")
recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    concordance_clean, models_reg_dt_gt, "classify_gt")
results_dict['dt_model_include_gt_excl_hpol_runs'] = models_dt_gt, models_reg_dt_gt
results_dict['dt_model_recall_precision_include_gt_excl_hpol_runs'] = recall_precision_gt
results_dict['dt_model_recall_precision_curve_include_gt_excl_hpol_runs'] = recall_precision_curve_gt


pickle.dump(results_dict, open(args.output_file, "wb"))
