import pandas as pd
import numpy as np


def read_hdf(fname: str):
    dfs = [pd.read_hdf(fname, key=f"chr{x}") for x in list(range(1, 23)) + ['X']]
    return pd.concat(dfs)


def classify_variants(vcf_concordance_df: pd.DataFrame, ignore_gt=True) -> pd.DataFrame:
    vcf_concordance_df['cat'] = None
    vcf_concordance_df.loc[vcf_concordance_df['indel'] == False, 'cat'] = 'SNP'
    vcf_concordance_df.loc[
        (vcf_concordance_df['indel'] == True) & (vcf_concordance_df['hmer_indel_length'] == 0) & (vcf_concordance_df['indel_length'] > 1), 'cat'] = 'NON-HMER'
    vcf_concordance_df.loc[
        (vcf_concordance_df['indel'] == True) & (vcf_concordance_df['hmer_indel_length'] == 0) & (vcf_concordance_df['indel_length'] == 1), 'cat'] = 'HMER 0/1'

    vcf_concordance_df.loc[(vcf_concordance_df['indel'] == True) & (vcf_concordance_df['hmer_indel_length'] > 0) & (
            vcf_concordance_df['hmer_indel_length'] <= 4), 'cat'] = 'HMER 2-4'
    vcf_concordance_df.loc[(vcf_concordance_df['indel'] == True) & (vcf_concordance_df['hmer_indel_length'] > 4) & (
            vcf_concordance_df['hmer_indel_length'] <= 8), 'cat'] = 'HMER 5-8'
    vcf_concordance_df.loc[(vcf_concordance_df['indel'] == True) & (vcf_concordance_df['hmer_indel_length'] > 8), 'cat'] = 'HMER >8'

    if ignore_gt:
        vcf_concordance_df['score'] = np.where(vcf_concordance_df['classify'] == 'fn', -1, vcf_concordance_df['tree_score'])
        vcf_concordance_df['label'] = np.where(vcf_concordance_df['classify'] == 'fp', 0, 1)
    else:
        vcf_concordance_df['score'] = np.where(vcf_concordance_df['classify_gt'] == 'fn', -1, vcf_concordance_df['tree_score'])
        vcf_concordance_df['label'] = np.where(vcf_concordance_df['classify_gt'] == 'fp', 0, 1)
    return vcf_concordance_df


def calc_performance(vcf_concordance_df: pd.DataFrame, cat: str):
    d = vcf_concordance_df[(vcf_concordance_df['cat'] == cat)].copy()
    d = d[['classify', 'score', 'label']].sort_values(by=['score'])
    num = len(d)
    numPos = sum(d['label'])
    numNeg = num - numPos
    d['fn'] = np.cumsum(d['label'])
    d['tp'] = numPos - (d['fn'])
    d['fp'] = numNeg - np.cumsum(1 - d['label'])
    d['recall'] = d['tp'] / (d['tp'] + d['fn'])
    d['precision'] = d['tp'] / (d['tp'] + d['fp'])
    d.loc[[d.index[-1]], 'precision'] = 1.0
    d['tresh'] = d.score.eq(d.score.shift(periods=-1))
    d = d[~d['tresh']]
    d['dist'] = ((1 - d['recall']) ** 2 + (1 - d['precision']) ** 2) ** (0.5)
    minDist = min(d['dist'])
    opt = d[d['dist'] == minDist]
    return d, opt, numPos, numNeg


def calculate_results(df):
    categories = ['SNP', 'NON-HMER', 'HMER 0/1', 'HMER 2-4', 'HMER 5-8', 'HMER >8']
    optRes = pd.DataFrame()
    for i, cat in enumerate(categories):
        d, opt, p, n = calc_performance(df, cat)
        row = pd.DataFrame({'pos': p,
                            'neg': n,
                            'tp': opt.tp[0],
                            'fp': opt.fp[0],
                            'fn': opt.fn[0],
                            'max recall': np.nan if d.empty else max(d.recall),
                            'initial_fp': np.nan if d.empty else max(d.fp),
                            'initial_tp': np.nan if d.empty else max(d.tp),
                            'recall': np.nan if opt.empty else opt.recall[0],
                            'precision': np.nan if opt.empty else opt.precision[0],
                            'thresh': '{:,.4f}'.format(opt.score[0])
                            }, index=[cat])
        optRes = pd.concat([optRes, row])

    # create INDEL ALL category
    indel_cats = ['NON-HMER', 'HMER 0/1', 'HMER 2-4', 'HMER 5-8', 'HMER >8']
    pos = 0
    neg = 0
    tp = 0
    fp = 0
    fn = 0
    initial_fp = 0
    initial_tp = 0
    for indel_cat in indel_cats:
        pos = pos + optRes.loc[indel_cat, 'pos']
        neg = neg + optRes.loc[indel_cat, 'neg']
        tp = tp + optRes.loc[indel_cat, 'tp']
        fp = fp + optRes.loc[indel_cat, 'fp']
        fn = fn + optRes.loc[indel_cat, 'fn']
        initial_fp = initial_fp + optRes.loc[indel_cat, 'initial_fp']
        initial_tp = initial_tp + optRes.loc[indel_cat, 'initial_tp']

    row = pd.DataFrame({'pos': pos,
                        'neg': neg,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'max recall': initial_tp / pos,
                        'initial_fp': initial_fp,
                        'initial_tp': initial_tp,
                        'recall': tp / pos,
                        'precision': tp / (fp + tp),
                        'thresh': np.nan
                        }, index=['INDEL_ALL'])
    optRes = pd.concat([optRes, row])
    return optRes


def apply_thresholds(df, thresholds):
    result_df = df.copy()
    blacklist_loc = result_df['filter'].str.contains('BLACKLIST')

    result_df['filter'] = 'PASS'
    result_df.loc[blacklist_loc, 'filter'] = 'BLACKLIST'
    thresholds = thresholds.loc[list(result_df['cat']), 'thresh']
    result_df['filter'] = result_df['filter'].where(
        pd.isnull(result_df['tree_score']) | (result_df['tree_score'] > np.array(thresholds.astype(np.float))),
        "LOW_SCORE")
    result_df.drop('tree_score', inplace=True, axis=1)
    return result_df


def filter_df(df):
    df = df[~((df.classify == "fp") & (df['filter'] == 'LOW_SCORE'))]
    return df
