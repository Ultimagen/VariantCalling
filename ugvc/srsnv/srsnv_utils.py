from __future__ import annotations

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


def get_balancing_sample_rate(motif_frequencies, min_quantile):
    return motif_frequencies.quantile(min_quantile) / motif_frequencies

def get_motif_sample_rate(df, motif, min_quantile):
    motif_freq = df[motif].value_counts() / df.shape[0]
    return get_balancing_sample_rate(motif_freq, min_quantile)

def get_motif_mul_sample_rate(df, motif1, motif2, min_quantile):
    motif_freq = (1000 * df[motif1] + df[motif2]).value_counts() / df.shape[0]
    return get_balancing_sample_rate(motif_freq, min_quantile)

def should_sample(label, sample_rate):
    return not label or random.random() <= sample_rate

def get_motif_freq_stds(df, min_quantile):
    stds = {}
    for motif in ("ref_alt_motif", "left_motif", "right_motif"):
        stds[motif] = get_motif_sample_rate(df, motif, min_quantile).var()
    return stds

def plot_motif_balance(df, title=None):
    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(15,4))
    if title:
        fig.suptitle(title)
    axs[0].set_ylabel('motif-enrichment')
    for i, motif in enumerate(['ref_alt_bases', 'ref_alt_motif', 'left_motif', 'right_motif']):
        axs[i].plot((df[motif].value_counts() /  df.shape[0]) / (1 / df[motif].value_counts().shape[0]) , '.')
        axs[i].axhline(y=1, color='red', linestyle='--')
        axs[i].set_xlabel(motif)

def balance_df_according_motif(df, motif, min_quantile):
    print(f"balance {motif}")
    motif_sample_rate = get_motif_sample_rate(df, motif, min_quantile)
    sample_motif = df.apply(
        lambda x: should_sample(x.label, motif_sample_rate[x[motif]]), axis=1
    )
    balanced = df[sample_motif]
    # plot_motif_balance(balanced)
    print(get_motif_freq_stds(balanced, min_quantile))
    print(f"motif balancing kept {balanced.shape[0] / df.shape[0]} of the data")
    return balanced

def precision_score_with_mask(y_pred: np.ndarray,  y_true: np.ndarray, mask: np.ndarray):
    if y_pred[mask].sum() == 0:
        return 1
    return precision_score(y_true[mask], y_pred[mask])

def recall_score_with_mask(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    if y_true[mask].sum() == 0:
        return 1
    return recall_score(y_true[mask], y_pred[mask])

def precision_recall_curve(score, max_score, y_true: np.ndarray, cumulative=False, apply_log_trans=True):
    precisions = []
    recalls = []
    fprs = []
    for i in range(max_score):
        if cumulative:
            mask = score.apply(np.floor) >= i
        else:
            mask = score.apply(np.floor) == i
        prediction = mask
        no_mask = score == score
        precisions.append(precision_score_with_mask(prediction, y_true, mask))
        recalls.append(recall_score_with_mask(prediction, y_true, no_mask))
        if precisions[-1] == np.nan:
            fprs.append(np.none)
        else:
            if apply_log_trans:
                if precisions[-1] == 1:
                        qual = max_score
                else:
                    qual = -10*np.log10(1 - precisions[-1])
                qual = min(qual, max_score)
                fprs.append(qual)
            else:
                fprs.append(precisions[-1])
    return fprs, recalls

def plot_precision_recall(lists, labels, max_score, log_scale=False, fs=14):
    for lst, label in zip(lists, labels):
        plt.plot(lst[0:max_score], '.-', label=label)
        if log_scale:
            plt.yscale('log')
        plt.xlabel('QUAL',fontsize=fs)
