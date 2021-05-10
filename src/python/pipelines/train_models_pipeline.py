import pathmagic
import python.pipelines.variant_filtering_utils as variant_filtering_utils
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import argparse
import pandas as pd
import numpy as np
import pickle
import sys

ap = argparse.ArgumentParser(prog="train_models_pipeline.py",
                             description="Train filtering models on the concordance file")
grp = ap.add_mutually_exclusive_group(required=True)
grp.add_argument("--input_file", help="Name of the input h5 file", type=str)
ap.add_argument("--output_file_prefix", help="Output .pkl file with models, .h5 file with results",
                type=str, required=True)
ap.add_argument("--mutect", required=False, action="store_true")
ap.add_argument("--evaluate_concordance", help="Should the results of the model be applied to the concordance dataframe",
                action="store_true")
ap.add_argument("--apply_model",
                help="If evaluate_concordance - which model should be applied", type=str, required=False)
ap.add_argument("--input_interval", help="bed file of intersected intervals from run_comparison pipeline",
                type=str, required=True)

args = ap.parse_args()
try:
    ## read all data besides concordance and input_args
    df = []
    with pd.HDFStore(args.input_file) as data:
        for k in data.keys():
            if (k != "/concordance") and (k != "/input_args"):
                h5_file = data.get(k)
                if not h5_file.empty:
                    df.append(h5_file)

    df = pd.concat(df, axis=0)
    if args.mutect:
        df['qual'] = df['tlod'].apply(lambda x: max(x) if type(x) == tuple else 50)*10
    df.loc[
        pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'
    #df = df.loc[(~np.isnan(df['qual']) & ~np.isinf(df['qual'])) | (df['classify'] =='fn')]
    df_clean = df[
        np.logical_not(df.close_to_hmer_run) & np.logical_not(df.inside_hmer_run)].copy()
    interval_size = vcf_pipeline_utils.bed_file_length(args.input_interval)
    # Unfiltered data
    # model_no_gt, recall_precision_no_gt = variant_filtering_utils.calculate_unfiltered_model(
    #     df.copy(), "classify")
    results_dict = {}
    # results_dict['unfiltered_ignore_gt_incl_hpol_runs'] = model_no_gt
    # results_dict[
    #     'unfiltered_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    #
    # model_gt, recall_precision_gt = variant_filtering_utils.calculate_unfiltered_model(
    #     df.copy(), "classify_gt")
    # results_dict['unfiltered_include_gt_incl_hpol_runs'] = model_gt
    # results_dict[
    #     'unfiltered_recall_precision_include_gt_incl_hpol_runs'] = recall_precision_gt
    #
    # model_no_gt, recall_precision_no_gt = variant_filtering_utils.calculate_unfiltered_model(
    #     df_clean.copy(), "classify")
    # results_dict['unfiltered_ignore_gt_excl_hpol_runs'] = model_no_gt
    # results_dict[
    #     'unfiltered_recall_precision_ignore_gt_excl_hpol_runs'] = recall_precision_no_gt
    #
    # model_gt, recall_precision_gt = variant_filtering_utils.calculate_unfiltered_model(
    #     df_clean.copy(), "classify_gt")
    # results_dict['unfiltered_include_gt_excl_hpol_runs'] = model_gt
    # results_dict[
    #     'unfiltered_recall_precision_include_gt_excl_hpol_runs'] = recall_precision_gt

    # Thresholding model
    models_thr_no_gt, models_reg_thr_no_gt, df_tmp = \
        variant_filtering_utils.train_threshold_models(
            df.copy(), interval_size, classify_column='classify')
    recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_thr_no_gt, "classify")
    recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_reg_thr_no_gt, "classify")

    results_dict[
        'threshold_model_ignore_gt_incl_hpol_runs'] = models_thr_no_gt, models_reg_thr_no_gt
    results_dict[
        'threshold_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    results_dict[
        'threshold_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt

    # models_thr_gt, models_reg_thr_gt, df_tmp = \
    #     variant_filtering_utils.train_threshold_models(
    #         df.copy(), interval_size, classify_column='classify_gt')
    # recall_precision_gt = variant_filtering_utils.test_decision_tree_model(
    #     df_tmp, models_thr_gt, "classify_gt")
    # recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    #     df_tmp, models_reg_thr_gt, "classify_gt")
    #
    # results_dict[
    #     'threshold_model_include_gt_incl_hpol_runs'] = models_thr_gt, models_reg_thr_gt
    # results_dict[
    #     'threshold_model_recall_precision_include_gt_incl_hpol_runs'] = recall_precision_gt
    # results_dict[
    #     'threshold_model_recall_precision_curve_include_gt_incl_hpol_runs'] = recall_precision_curve_gt
    #
    # models_thr_no_gt, models_reg_thr_no_gt, df_clean_tmp = \
    #     variant_filtering_utils.train_threshold_models(
    #         df_clean.copy(), interval_size, classify_column='classify')
    # recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
    #     df_clean_tmp, models_thr_no_gt, "classify")
    # recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    #     df_clean_tmp, models_reg_thr_no_gt, "classify")
    #
    # results_dict[
    #     'threshold_model_ignore_gt_excl_hpol_runs'] = models_thr_no_gt, models_reg_thr_no_gt
    # results_dict[
    #     'threshold_model_recall_precision_ignore_gt_excl_hpol_runs'] = recall_precision_no_gt
    # results_dict[
    #     'threshold_model_recall_precision_curve_ignore_gt_excl_hpol_runs'] = recall_precision_curve_no_gt
    # models_thr_gt, models_reg_thr_gt, df_clean_tmp = \
    #     variant_filtering_utils.train_threshold_models(
    #         df_clean.copy(), interval_size, classify_column='classify_gt')
    # recall_precision_gt = variant_filtering_utils.test_decision_tree_model(
    #     df_clean_tmp, models_thr_gt, "classify_gt")
    # recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    #     df_clean_tmp, models_reg_thr_gt, "classify_gt")
    #
    # results_dict[
    #     'threshold_model_include_gt_excl_hpol_runs'] = models_thr_gt, models_reg_thr_gt
    # results_dict[
    #     'threshold_model_recall_precision_include_gt_excl_hpol_runs'] = recall_precision_gt
    # results_dict[
    #     'threshold_model_recall_precision_curve_include_gt_excl_hpol_runs'] = recall_precision_curve_gt

    # Decision tree models
    models_dt_no_gt, models_reg_dt_no_gt, df_tmp = \
        variant_filtering_utils.train_decision_tree_model(df.copy(),
                                                          classify_column='classify', interval_size=interval_size)
    recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_dt_no_gt, "classify")
    recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_reg_dt_no_gt, "classify")

    ### check on trained data
    df_tmp['test_train_split'] = np.logical_not(df_tmp['test_train_split'])
    trained_recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_dt_no_gt, "classify")
    trained_recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_reg_dt_no_gt, "classify")
    df_tmp['test_train_split'] = np.logical_not(df_tmp['test_train_split'])

    results_dict[
        'dt_model_ignore_gt_incl_hpol_runs'] = models_dt_no_gt, models_reg_dt_no_gt
    results_dict[
        'dt_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    results_dict[
        'dt_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt

    results_dict[
        'trained_dt_model_recall_precision_ignore_gt_incl_hpol_runs'] = trained_recall_precision_no_gt
    results_dict[
        'trained_dt_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = trained_recall_precision_curve_no_gt

    # models_dt_gt, models_reg_dt_gt, df_tmp = \
    #     variant_filtering_utils.train_decision_tree_model(df.copy(),
    #                                                       classify_column='classify_gt', interval_size=interval_size)
    # recall_precision_gt = variant_filtering_utils.test_decision_tree_model(
    #     df_tmp, models_dt_gt, "classify_gt")
    # recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    #     df_tmp, models_reg_dt_gt, "classify_gt")
    #
    # results_dict[
    #     'dt_model_include_gt_incl_hpol_runs'] = models_dt_gt, models_reg_dt_gt
    # results_dict[
    #     'dt_model_recall_precision_include_gt_incl_hpol_runs'] = recall_precision_gt
    # results_dict[
    #     'dt_model_recall_precision_curve_include_gt_incl_hpol_runs'] = recall_precision_curve_gt

    # models_dt_no_gt, models_reg_dt_no_gt, df_clean_tmp = \
    #     variant_filtering_utils.train_decision_tree_model(df_clean.copy(),
    #                                                       classify_column='classify', interval_size=interval_size)
    # recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
    #     df_clean_tmp, models_dt_no_gt, "classify")
    # recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    #     df_clean_tmp, models_reg_dt_no_gt, "classify")
    #
    # results_dict[
    #     'dt_model_ignore_gt_excl_hpol_runs'] = models_dt_no_gt, models_reg_dt_no_gt
    # results_dict[
    #     'dt_model_recall_precision_ignore_gt_excl_hpol_runs'] = recall_precision_no_gt
    # results_dict[
    #     'dt_model_recall_precision_curve_ignore_gt_excl_hpol_runs'] = recall_precision_curve_no_gt
    #
    # models_dt_gt, models_reg_dt_gt, df_clean_tmp = \
    #     variant_filtering_utils.train_decision_tree_model(df_clean.copy(),
    #                                                       classify_column='classify_gt', interval_size=interval_size)
    # recall_precision_gt = variant_filtering_utils.test_decision_tree_model(
    #     df_clean_tmp, models_dt_gt, "classify_gt")
    # recall_precision_curve_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
    #     df_clean_tmp, models_reg_dt_gt, "classify_gt")
    # results_dict[
    #     'dt_model_include_gt_excl_hpol_runs'] = models_dt_gt, models_reg_dt_gt
    # results_dict[
    #     'dt_model_recall_precision_include_gt_excl_hpol_runs'] = recall_precision_gt
    # results_dict[
    #     'dt_model_recall_precision_curve_include_gt_excl_hpol_runs'] = recall_precision_curve_gt

    pickle.dump(results_dict, open(args.output_file_prefix + ".pkl", "wb"))

    optdict = {}
    prcdict = {}
    for model in [ 'dt', 'trained_dt']:
        for gt in ['ignore']:
            for hpol in ['incl']:
                name_optimum = f'{model}_model_recall_precision_{gt}_gt_{hpol}_hpol_runs'
                optdict[name_optimum] = results_dict[name_optimum]
                prcdict[name_optimum] = results_dict[name_optimum.replace(
                    "recall_precision", "recall_precision_curve")]

    results_vals = (pd.DataFrame(optdict)).unstack().reset_index()
    results_vals.columns = ['model', 'category', 'tmp']
    results_vals.loc[pd.isnull(results_vals['tmp']), 'tmp'] = [
        (np.nan, np.nan)]
    results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[0])
    results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[1])
    results_vals['f1'] = results_vals['tmp'].apply(lambda x: x[2])
    results_vals.drop('tmp', axis=1, inplace=True)

    results_vals.to_hdf(args.output_file_prefix + '.h5',
                        key="optimal_recall_precision")

    results_vals = (pd.DataFrame(prcdict)).unstack().reset_index()
    results_vals.columns = ['model', 'category', 'tmp']
    results_vals.loc[pd.isnull(results_vals['tmp']), 'tmp'] = [
        np.zeros((0, 2))]
    results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[:, 0])
    results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[:, 1])
    #results_vals['f1'] = results_vals['tmp'].apply(lambda x: x[:,2])
    results_vals.drop('tmp', axis=1, inplace=True)

    results_vals.to_hdf(args.output_file_prefix + ".h5",
                        key="recall_precision_curve")

    if args.evaluate_concordance:

        concordance = pd.read_hdf(args.input_file, "concordance")

        if args.mutect:
            concordance['qual'] = concordance['tlod'].apply(lambda x: max(x) if type(x) == tuple else 50) * 10
        concordance.loc[
            pd.isnull(concordance['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'

        models = results_dict[args.apply_model]
        model_clsf = models[0]
        model_scor = models[1]

        print("Applying classifier", flush=True, file=sys.stderr)
        predictions = model_clsf.predict(
            variant_filtering_utils.add_grouping_column(concordance,
                                                        variant_filtering_utils.get_training_selection_functions(),
                                                        "group"))
        print("Applying regressor", flush=True, file=sys.stderr)

        predictions_score = model_scor.predict(
            variant_filtering_utils.add_grouping_column(concordance,
                                                        variant_filtering_utils.get_training_selection_functions(),
                                                        "group"))



        concordance['prediction'] = predictions
        concordance['tree_score'] = predictions_score
        concordance.to_hdf(args.output_file_prefix +
                           ".h5", key="scored_concordance")
    print("Model training run: success", file=sys.stderr, flush=True)

except Exception as err:
    exc_info = sys.exc_info()
    print(*exc_info, file=sys.stderr, flush=True)
    print("Model training run: failed", file=sys.stderr, flush=True)
    raise(err)
