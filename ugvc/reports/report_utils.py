import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nexusplt as nxp
from ugvc.utils.stats_utils import get_f1, get_precision, get_recall
from configparser import ConfigParser


def parse_config(config_file):
    parser = ConfigParser()
    parser.read(config_file)
    param_names = ['run_id',
                   'pipeline_version',
                   'h5_concordance_file',
                   'model_name_with_gt',
                   'model_name_without_gt',
                   'model_pkl_with_gt',
                   'model_pkl_without_gt'
                   ]
    parameters = {p: parser.get('VarReport', p) for p in param_names}
    parameters['reference_version'] = parser.get('VarReport', 'reference_version', fallback='hg38')

    # Optional parameters
    parameters['truth_sample_name'] = parser.get('VarReport', 'truth_sample_name', fallback='NA')
    parameters['sources'] = {'Trained wo gt': (parameters['h5_concordance_file'], "concordance")}
    parameters['image_prefix'] = parser.get('VarReport', 'image_output_prefix',
                                            fallback=parameters['run_id'] + '.vars') + '.'
    parameters['h5outfile'] = parser.get('VarReport', 'h5_output', fallback='var_report.h5')
    parameters['trained_w_gt'] = parser.get('VarReport', 'h5_model_file', fallback=None)

    return parameters, param_names


class ShortReportUtils:

    def __init__(self, image_dir, image_prefix):

        self.image_dir = image_dir
        self.image_prefix = image_prefix

    def plot_performance(self,
                         perf_curve,
                         opt_res,
                         categories,
                         sources: dict,
                         ext=None,
                         img=None,
                         legend=None,
                         opt_res_sec=None):
        n = len(categories)
        fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
        # fig, ax = plt.subplots(1,ncol) #figsize=(4*nrow,4*ncol)
        # ax=ax.flatten()

        col = ['r', 'b', 'g', 'm', 'k']

        for i, cat in enumerate(categories):
            for j, s in enumerate(sources):
                perf = perf_curve[s][cat]
                opt = opt_res[s][cat]
                if opt_res_sec is not None:
                    opt_sec = opt_res_sec[s][cat]
                if not perf.empty:
                    ax[i].plot(perf.recall, perf.precision, '-', label=s, color=col[j])
                    ax[i].plot(opt.get('recall'), opt.get('precision'), 'o', color=col[j])
                    if opt_res_sec is not None:
                        ax[i].plot(opt_sec.get('recall'), opt_sec.get('precision'), 'o', color="black")
                title = cat if ext is None else '{0} ({1})'.format(cat, ext)
                ax[i].set_title(title)
                ax[i].set_xlabel("Recall")
                ax[i].set_xlim([0.4, 1])
                ax[i].set_ylim([0.4, 1])
                ax[i].grid(True)

        ax[0].set_ylabel("Precision")
        if legend:
            ax[0].legend(loc='lower left')
        if img:
            nxp.save(fig, self.image_prefix + img, 'png', outdir=self.image_dir)

    def get_performance(self, data, categories, sources):
        opt_tab = {}
        opt_res = {}
        perf_curve = {}
        for s in sources:
            opt_tab[s] = pd.DataFrame()
            opt_res[s] = {}
            perf_curve[s] = {}

            for i, cat in enumerate(categories):
                d = self.__filter_by_category(data[s], cat)
                perf, opt, pos, neg = self.__calc_performance(d)
                perf_curve[s][cat] = perf
                opt_res[s][cat] = opt

                row = pd.DataFrame({'# pos': pos,
                                    '# neg': neg,
                                    'max recall': np.nan if perf.empty else max(perf.recall),
                                    'recall': np.nan if perf.empty else opt.get('recall'),
                                    'precision': np.nan if perf.empty else opt.get('precision'),
                                    'F1': np.nan if perf.empty else opt.get('f1')
                                    }, index=[cat])
                opt_tab[s] = pd.concat([opt_tab[s], row])

        return opt_tab, opt_res, perf_curve

    @staticmethod
    def has_sec(x):
        res = False
        if x is not None:
            if x == x:
                if "SEC" in x:
                    res = True
        return res

    @staticmethod
    def __calc_performance(data):
        classify_col = 'classify_gt'
        d = data.copy()

        # Change tree_score such that PASS will get high score values. FNs will be the lowest
        score_pass = d.query('filter=="PASS" & tree_score==tree_score').head(20)['tree_score'].mean()
        score_not_pass = d.query('filter!="PASS" & tree_score==tree_score').head(20)['tree_score'].mean()
        dir_switch = 1 if score_pass > score_not_pass else -1
        score = d['tree_score'] * dir_switch
        score = score - score.min()
        d['tree_score'] = np.where(d[classify_col] == 'fn', -1, score)

        # Calculate precision and recall continuously along the tree_score values
        d = d[[classify_col, 'tree_score', 'filter']].sort_values(by=['tree_score'])

        d['label'] = np.where(d[classify_col] == 'fp', 0, 1)

        d.loc[d[classify_col] == 'fn', 'filter'] = 'MISS'

        num = len(d)
        num_pos = sum(d['label'])
        num_neg = num - num_pos
        if num < 10:
            return pd.DataFrame(), None, num_pos, num_neg

        d['fn'] = np.cumsum(d['label'])
        d['tp'] = num_pos - (d['fn'])
        d['fp'] = num_neg - np.cumsum(1 - d['label'])

        d['recall'] = get_recall(d['fn'], d['tp'])
        d['precision'] = get_precision(d['fp'], d['tp'])
        d['f1'] = get_f1(d['precision'], d['recall'])

        d['mask'] = ((d['tp'] + d['fn']) >= 20) & ((d['tp'] + d['fp']) >= 20) & (d['tree_score'] >= 0)
        if len(d[d['mask']]) == 0:
            return pd.DataFrame(), None, num_pos, num_neg

        # Calculate the precision and recall as ouputted by the model (based on the FILTER column)
        d['class'] = np.where(d['label'] == 0, 'FP', 'FN')
        d.loc[(d['label'] == 1) & (d['filter'] == 'PASS'), 'class'] = 'TP'
        d.loc[(d['label'] == 0) & (d['filter'] != 'PASS'), 'class'] = 'TN'

        fn = len(d[d['class'] == 'FN'])
        tp = len(d[d['class'] == 'TP'])
        fp = len(d[d['class'] == 'FP'])

        recall = tp / (tp + fn) if (tp + fn > 0) else np.nan
        precision = tp / (tp + fp) if (tp + fp > 0) else np.nan
        max_recall = 1 - len(d[d['filter'] == 'MISS']) / num_pos

        f1 = tp / (tp + 0.5 * fn + 0.5 * fp)

        return (d[['recall', 'precision']][d['mask']],
                dict({'recall': recall, 'precision': precision, 'f1': f1}),
                num_pos, num_neg)

    @staticmethod
    def __filter_by_category(data, cat):
        if cat == 'SNP':
            return data[data['indel'] == False]
        if cat == 'Indel':
            return data[data['indel'] == True]
        elif cat == 'non-hmer Indel':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] == 0) & (data['indel_length'] > 0)]
        elif cat == 'non-hmer Indel w/o LCR':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] == 0) & (data['indel_length'] > 0) &
                        (~data['LCR'])]
        elif cat == 'hmer Indel <=4':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] > 0) & (data['hmer_indel_length'] <= 4)]
        elif cat == 'hmer Indel >4,<=8':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] > 4) & (data['hmer_indel_length'] <= 8)]
        elif cat == 'hmer Indel >8,<=10':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] > 8) & (data['hmer_indel_length'] <= 10)]
        elif cat == 'hmer Indel >10,<=14':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] > 10) & (data['hmer_indel_length'] <= 14)]
        elif cat == 'hmer Indel >15,<=19':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] > 14) & (data['hmer_indel_length'] <= 19)]
        elif cat == 'hmer Indel >=20':
            return data[(data['indel'] == True) & (data['hmer_indel_length'] >= 20)]
        for i in range(1, 10):
            if cat == 'hmer Indel {0:d}'.format(i):
                return data[(data['indel'] == True) & (data['hmer_indel_length'] == i)]
        return None
