#!/env/python
import pathmagic
import python.pipelines.variant_filtering_utils as variant_filtering_utils
import argparse
import pandas as pd
import pickle
import numpy as np

ap = argparse.ArgumentParser(prog="evaluate_concordance.py",
                             description="Calculate precision and recall for compared HDF5 ")
ap.add_argument("--input_file", help="Name of the input h5 file", type=str)
ap.add_argument("--output_file", help="Output h5 file", type=str, required=True)
args = ap.parse_args()

concordance = pd.read_hdf(args.input_file, key="concordance")
assert 'tree_score' in concordance.columns, "Input concordance file should be after applying a model"

concordance.loc[pd.isnull(concordance['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'


concordance['group'] = 'all'
concordance['test_train_split'] = False
concordance['group_testing'] = variant_filtering_utils.add_grouping_column(
    concordance, variant_filtering_utils.get_testing_selection_functions(), "group_testing")

trivial_classifier = variant_filtering_utils.SingleTrivialClassifierModel()
trivial_regressor = variant_filtering_utils.SingleTrivialRegressorModel()

trivial_classifier_set = variant_filtering_utils.MaskedHierarchicalModel(
    'classifier', 'group', {'all': trivial_classifier})
trivial_regressor_set = variant_filtering_utils.MaskedHierarchicalModel(
    'regressor', 'group', {'all': trivial_regressor})

recall_precision_dict = {}
recall_precision_curve_dict = {}
for exclude_hpols in [False, True]:
    for ignore_gt in [False, True]:
        name = 'untrained_%s_%s' % (['include_gt', 'ignore_gt'][ignore_gt],
                                    ['incl_hpol_runs', 'excl_hpol_runs'][exclude_hpols])

        if exclude_hpols:
            is_hpol_run = concordance['filter'].apply(lambda x: 'HPOL_RUN' in x)
            concordance_filtered = concordance[~is_hpol_run].copy()
        else:
            concordance_filtered = concordance.copy()

        classify_column = ['classify_gt', 'classify'][ignore_gt]
        recall_precision = variant_filtering_utils.test_decision_tree_model(
            concordance_filtered, trivial_classifier_set, classify_column)
        recall_precision_dict[name] = recall_precision

        recall_precision_curve = variant_filtering_utils.get_decision_tree_precision_recall_curve(
            concordance_filtered, trivial_regressor_set, classify_column)
        recall_precision_curve_dict[name] = recall_precision_curve

results_vals = (pd.DataFrame(recall_precision_dict)).unstack().reset_index()
results_vals.columns = ['model', 'category','tmp']
results_vals.loc[pd.isnull(results_vals['tmp']), 'tmp'] = [(np.nan, np.nan)]
results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[0])
results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[1])
results_vals.drop('tmp',axis=1,inplace=True)

results_vals.to_hdf(args.output_file, key="optimal_recall_precision")

results_vals = (pd.DataFrame(recall_precision_curve_dict)).unstack().reset_index()
results_vals.columns = ['model', 'category','tmp']
results_vals.loc[pd.isnull(results_vals['tmp']),'tmp'] = [np.zeros((0,2))]
results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[:,0])
results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[:,1])

results_vals.drop('tmp',axis=1,inplace=True)

results_vals.to_hdf(args.output_file, key="recall_precision_curve")

