import pathmagic
import python.pipelines.variant_filtering_utils as variant_filtering_utils
import python.pipelines.vcf_pipeline_utils as vcf_pipeline_utils
import python.vcftools as vcftools
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
import logging

ap = argparse.ArgumentParser(prog="train_models_pipeline.py",
                             description="Train filtering models on the concordance file")

ap.add_argument("--input_file", help="Name of the input h5/vcf file. h5 is output of comparison", type=str)
ap.add_argument("--blacklist", help="blacklist file by which we decide variants as FP", type=str, required=False)
ap.add_argument("--output_file_prefix", help="Output .pkl file with models, .h5 file with results",
                type=str, required=True)
ap.add_argument("--mutect", required=False, action="store_true")
ap.add_argument("--evaluate_concordance", help="Should the results of the model be applied to the concordance dataframe",
                action="store_true")
ap.add_argument("--apply_model",
                help="If evaluate_concordance - which model should be applied", type=str, required='--evaluate_concordance' in sys.argv)
ap.add_argument("--input_interval", help="bed file of intersected intervals from run_comparison pipeline",
                type=str, required=False)
ap.add_argument("--list_of_contigs_to_read", nargs='*', help="List of contigs to read from the DF", default=[])
ap.add_argument("--reference", help='Reference genome',
                required=True, type=str)
ap.add_argument("--runs_intervals", help='Runs intervals (bed/interval_list)',
                required=False, type=str, default=None)
ap.add_argument("--annotate_intervals", help='interval files for annotation (multiple possible)', required=False,
                type=str, default=None, action='append')
ap.add_argument("--exome_weight", help='weight of exome variants in comparison to whole genome variant',
                type=int, default=1)
ap.add_argument("--flow_order",
                help="Sequencing flow order (4 cycle)", required=False, default="TGCA")
ap.add_argument("--exome_weight_annotation", help='annotation name by which we decide the weight of exome variants',
                type=str)
ap.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

args = ap.parse_args()

logging.basicConfig(level=getattr(logging, args.verbosity),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__ if __name__ != "__main__" else "train_model_pipeline")

try:
    if args.input_file.endswith('h5'):
        assert args.input_interval,\
            "--input_interval is required when input file type is h5"
    if args.input_file.endswith('vcf.gz'):
        assert args.blacklist,\
            "--blacklist is required when input file type is vcf.gz"
        assert args.reference,\
            "--reference is required when input file type is vcf.gz"
        assert args.runs_intervals,\
            "--runs_intervals is required when input file type is vcf.gz"
except AssertionError as af:
    logger.error(str(af))
    raise af

try:
    with_dbsnp_bl = args.input_file.endswith('vcf.gz')
    if with_dbsnp_bl:
        df = vcftools.get_vcf_df(args.input_file)

    else:
        ## read all data besides concordance and input_args or as defined in list_of_contigs_to_read
        df = []
        annots = []
        with pd.HDFStore(args.input_file) as data:
            for k in data.keys():
                if (k != "/concordance") and (k != "/input_args") and \
                        (args.list_of_contigs_to_read == [] or k[1:] in args.list_of_contigs_to_read):
                    h5_file = data.get(k)
                    if not h5_file.empty:
                        df.append(h5_file)

        df = pd.concat(df, axis=0)

    df, annots = vcf_pipeline_utils.annotate_concordance(df, args.reference,
                                                         runfile=args.runs_intervals,
                                                         flow_order=args.flow_order,
                                                         annotate_intervals=args.annotate_intervals)

    if args.mutect:
        df['qual'] = df['tlod'].apply(lambda x: max(x) if type(x) == tuple else 50)*10
    df.loc[
        pd.isnull(df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'


    results_dict = {}

    if with_dbsnp_bl:
        blacklist = pd.read_hdf(args.blacklist,'blacklist')
        df = df.merge(blacklist, left_index=True, right_index=True, how='left')

        df['bl_classify'] = 'unknown'
        df['bl_classify'].loc[df['bl'] == True] = 'fp'
        df['bl_classify'].loc[~df['id'].isna()] = 'tp'
        df = df[df['bl_classify'] != 'unknown']
        # Decision tree models
        classify_clm = 'bl_classify'
        interval_size = None
    else:
        classify_clm = 'classify'
        interval_size = vcf_pipeline_utils.bed_file_length(args.input_interval)



    # Thresholding model
    models_thr_no_gt, models_reg_thr_no_gt, df_tmp = \
        variant_filtering_utils.train_threshold_models(
            df.copy(), interval_size, classify_column=classify_clm, annots=annots)
    recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_thr_no_gt, classify_column=classify_clm)
    recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_reg_thr_no_gt, classify_column=classify_clm)

    results_dict[
        'threshold_model_ignore_gt_incl_hpol_runs'] = models_thr_no_gt, models_reg_thr_no_gt
    results_dict[
        'threshold_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    results_dict[
        'threshold_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt

    # decision tree model
    models_dt_no_gt, models_reg_dt_no_gt, df_tmp = \
        variant_filtering_utils.train_model_wrapper(df.copy(),
                                                    classify_column=classify_clm,
                                                    interval_size=interval_size,
                                                    train_function=variant_filtering_utils.train_model_DT,
                                                    model_name="Decision tree",
                                                    annots=annots,
                                                    exome_weight=args.exome_weight,
                                                    exome_weight_annotation=args.exome_weight_annotation,
                                                    use_train_test_split=True)
    # if with_dbsnp_bl:
    #     df_tmp['test_train_split'] = False
    recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_dt_no_gt, classify_clm)
    recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_dt_no_gt, classify_clm, proba=True)

    results_dict[
        'dt_model_ignore_gt_incl_hpol_runs'] = models_dt_no_gt, models_reg_dt_no_gt
    results_dict[
        'dt_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    results_dict[
        'dt_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt

    # NN model
    models_nn_no_gt, models_reg_nn_no_gt, df_tmp = \
        variant_filtering_utils.train_model_wrapper(df.copy(),
                                                    classify_column=classify_clm,
                                                    interval_size=interval_size,
                                                    train_function=variant_filtering_utils.train_model_NN,
                                                    model_name="Neural network",
                                                    annots=annots,
                                                    exome_weight=args.exome_weight,
                                                    exome_weight_annotation=args.exome_weight_annotation,
                                                    use_train_test_split=True)
    if with_dbsnp_bl:
        df_tmp['test_train_split'] = False
    recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_nn_no_gt, classify_clm)
    recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_nn_no_gt, classify_clm, proba=True)

    results_dict[
        'nn_model_ignore_gt_incl_hpol_runs'] = models_nn_no_gt, models_reg_nn_no_gt
    results_dict[
        'nn_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    results_dict[
        'nn_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt

    # RF model
    models_rf_no_gt, models_reg_rf_no_gt, df_tmp = \
        variant_filtering_utils.train_model_wrapper(df.copy(),
                                                    classify_column=classify_clm,
                                                    interval_size=interval_size,
                                                    train_function=variant_filtering_utils.train_model_RF,
                                                    model_name="Random forest",
                                                    annots=annots,
                                                    exome_weight=args.exome_weight,
                                                    exome_weight_annotation=args.exome_weight_annotation,
                                                    use_train_test_split=True)
    # if with_dbsnp_bl:
    #     df_tmp['test_train_split'] = False


    recall_precision_no_gt = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_rf_no_gt, classify_clm, proba=True)
    recall_precision_curve_no_gt = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_rf_no_gt, classify_clm, proba=True)

    df_tmp["test_train_split"] = ~df_tmp["test_train_split"]
    recall_precision_no_gt_train = variant_filtering_utils.test_decision_tree_model(
        df_tmp, models_rf_no_gt, classify_clm, proba=True)
    recall_precision_curve_no_gt_train = variant_filtering_utils.get_decision_tree_precision_recall_curve(
        df_tmp, models_rf_no_gt, classify_clm, proba=True)




    results_dict[
        'rf_model_ignore_gt_incl_hpol_runs'] = models_rf_no_gt, models_reg_rf_no_gt
    results_dict[
        'rf_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt
    results_dict[
        'rf_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt
    results_dict[
        'rf_train_model_recall_precision_ignore_gt_incl_hpol_runs'] = recall_precision_no_gt_train
    results_dict[
        'rf_train_model_recall_precision_curve_ignore_gt_incl_hpol_runs'] = recall_precision_curve_no_gt_train



    pickle.dump(results_dict, open(args.output_file_prefix + ".pkl", "wb"))

    optdict = {}
    prcdict = {}
    for m in ['dt','nn','threshold','rf','rf_train']:
        name_optimum = f'{m}_model_recall_precision_ignore_gt_incl_hpol_runs'
        optdict[name_optimum] = results_dict[name_optimum]
        prcdict[name_optimum] = results_dict[name_optimum.replace(
            "recall_precision", "recall_precision_curve")]

    results_vals = (pd.DataFrame(optdict)).unstack().reset_index()
    results_vals.columns = ['model', 'category', 'tmp']
    results_vals.loc[pd.isnull(results_vals['tmp']), 'tmp'] = [
        (np.nan, np.nan, np.nan)]
    results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[0])
    results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[1])
    results_vals['f1'] = results_vals['tmp'].apply(lambda x: x[2])
    results_vals.drop('tmp', axis=1, inplace=True)

    results_vals.to_hdf(args.output_file_prefix + '.h5',
                        key="optimal_recall_precision")

    results_vals = (pd.DataFrame(prcdict)).unstack().reset_index()
    results_vals.columns = ['model', 'category', 'tmp']
    results_vals.loc[pd.isnull(results_vals['tmp']), 'tmp'] = [
        np.zeros((0, 3))]
    results_vals['recall'] = results_vals['tmp'].apply(lambda x: x[:, 0])
    results_vals['precision'] = results_vals['tmp'].apply(lambda x: x[:, 1])
    results_vals['f1'] = results_vals['tmp'].apply(lambda x: x[:,2])
    results_vals.drop('tmp', axis=1, inplace=True)

    results_vals.to_hdf(args.output_file_prefix + ".h5",
                        key="recall_precision_curve")

    if args.evaluate_concordance:
        if with_dbsnp_bl:
            calls_df = vcftools.get_vcf_df(args.input_file, chromosome='chr9')
        else:
            calls_df = pd.read_hdf(args.input_file, "concordance")
        calls_df, _ = vcf_pipeline_utils.annotate_concordance(calls_df, args.reference,
                                                              runfile=args.runs_intervals,
                                                              annotate_intervals=args.annotate_intervals)
        if args.mutect:
            calls_df['qual'] = calls_df['tlod'].apply(lambda x: max(x) if type(x) == tuple else 50) * 10
        calls_df.loc[
            pd.isnull(calls_df['hmer_indel_nuc']), "hmer_indel_nuc"] = 'N'

        models = results_dict[args.apply_model]
        model_clsf = models[0]
        model_scor = models[1]

        print("Applying classifier", flush=True, file=sys.stderr)
        predictions = model_clsf.predict(
            variant_filtering_utils.add_grouping_column(calls_df,
                                                        variant_filtering_utils.get_training_selection_functions(),
                                                        "group"))
        print("Applying regressor", flush=True, file=sys.stderr)

        predictions_score = model_scor.predict(
            variant_filtering_utils.add_grouping_column(calls_df,
                                                        variant_filtering_utils.get_training_selection_functions(),
                                                        "group"))

        calls_df['prediction'] = predictions
        calls_df['tree_score'] = predictions_score
        # In case we already have filter column, reset the PASS,
        # Then, by the prediction of the model we decide whether the filter column is PASS or LOW_SCORE
        calls_df['filter'] = calls_df['filter'].apply(lambda x: x.replace('PASS;', '')).\
            apply(lambda x: x.replace(';PASS', '')).\
            apply(lambda x: x.replace('PASS', ''))
        calls_df.loc[calls_df['prediction']=='fp','filter'] = \
            calls_df.loc[calls_df['prediction']=='fp','filter'].apply(
            lambda x: 'LOW_SCORE' if x=='' else x+';LOW_SCORE')
        calls_df.loc[calls_df['prediction'] == 'tp','filter'] = \
            calls_df.loc[calls_df['prediction'] == 'tp','filter'].apply(
            lambda x: 'PASS' if x == '' else x + ';PASS')
        calls_df.to_hdf(args.output_file_prefix +
                           ".h5", key="scored_concordance")
    print("Model training run: success", file=sys.stderr, flush=True)

except Exception as err:
    exc_info = sys.exc_info()
    print(*exc_info, file=sys.stderr, flush=True)
    print("Model training run: failed", file=sys.stderr, flush=True)
    raise(err)
