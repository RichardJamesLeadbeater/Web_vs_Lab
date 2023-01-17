import pandas as pd
from scipy import stats as st
import numpy as np
from collections import namedtuple as ntup
import os
from mytools.dframe_tools import to_csv_pkl


def mean_diff(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    diff = (x - y)
    Output = ntup('output', ['mean_diff', 'stdev'], defaults=[np.mean(diff), np.std(diff, ddof=1)])
    return Output()


def perform_stats(a, b, dicty, ratio, ntail='two-sided'):
    dicty['ratio'].append(ratio)

    if isinstance(a, np.ndarray):
        pass
    else:
        a = np.asarray(a[measure].to_list())
        b = np.asarray(b[measure].to_list())
    dicty['shapiro (p)'].append(st.shapiro(a - b)[1])
    paired_t = st.ttest_rel(a, b, alternative=ntail)
    dicty['paired_t (t)'].append(paired_t[0])
    dicty['(p)'].append(paired_t[1])
    dicty['cohens_d'].append(paired_cohens_d(a, b))


def add_descriptive_to_dicty(_expname, array, dicty, ratio, geo_mean=False, antilog=False):
    dicty['ratio'].append(ratio)
    dicty['task'].append(_expname)
    if geo_mean:
        logarray = np.log(array)
        dicty['mean'].append(np.exp(np.mean(logarray)))
        dicty['sd'].append(np.exp(np.std(logarray, ddof=1)))
        dicty['sem'].append(np.exp(st.stats.sem(logarray, ddof=1)))
    elif antilog:
        dicty['mean'].append(np.exp(np.mean(array)))
        dicty['sd'].append(np.exp(np.std(array, ddof=1)))
        dicty['sem'].append(np.exp(st.stats.sem(array, ddof=1)))
    else:
        dicty['mean'].append(np.mean(array))
        dicty['sd'].append(np.std(array, ddof=1))
        dicty['sem'].append(st.stats.sem(array, ddof=1))


def paired_cohens_d(x, y):
    diff = mean_diff(x, y)
    return diff.mean_diff / diff.stdev


# init paths
og_path = os.getcwd()
sum_path = os.path.join(og_path, 'data', 'summary')
stats_path = os.path.join(og_path, 'descriptive statistics')
if not os.path.exists(stats_path):
    os.makedirs(stats_path)

# common keys as vars
measure = 'threshold'
ori = 'ori'
exp = 'task'
env = 'enviro'
p = 'observer'

# SUMMARY STATS #
summary = pd.read_csv(os.path.join(sum_path, f"summary_data.csv"))

samples = ['allweb', 'web', 'lab']
wvl_ps = pd.read_csv('subsample_ids.csv')['p_id'].to_list()
# wvl_ps = [f"p{i}" for i in range(13) if i != 0]
for isample in samples:
    summary_stats = {'sample': [], ori: [], exp: [], 'n': [], 'mean': [], 'stdev': [], 'sem': [], '95%CI': [],
                     'median': [], 'iqr': [], 'top95': [], 'range': []}
    if isample == 'allweb':
        isumdata = summary[summary[env] == 'web']
    elif isample == 'lab':
        isumdata = summary[summary[env] == 'lab']
    elif isample == 'web':
        isumdata = pd.concat([summary[summary[p] == i] for i in wvl_ps])
        isumdata = isumdata[isumdata[env] == 'web']
    if isumdata.empty:
        print(f"\n>>> DFRAME EMPTY FOR {isample}")
    for iori in summary[ori].unique():
        for i_task in summary[exp].unique():
            idata = isumdata[(isumdata[ori] == iori) &
                             (isumdata[exp] == i_task)]
            if len(isumdata) == 0:
                continue
            summary_stats['sample'].append(isample)
            summary_stats['ori'].append(iori)
            summary_stats[exp].append(i_task)
            summary_stats['n'].append(len(idata))
            summary_stats['mean'].append(idata['threshold'].mean())
            summary_stats['stdev'].append(idata['threshold'].std())
            summary_stats['sem'].append(st.sem(idata['threshold']))
            i_CIs = st.t.interval(0.95, len(idata) - 1, loc=idata['threshold'].mean(),
                                  scale=st.sem(idata['threshold']))  # lower and upper bound of conf interval
            summary_stats['95%CI'].append([np.round(ci, 2) for ci in i_CIs])
            summary_stats['median'].append(idata['threshold'].median())
            iqr = np.percentile(idata['threshold'].to_numpy(), [25, 75])
            summary_stats['iqr'].append([np.round(r, 2) for r in iqr])  # interquartile range
            top95 = np.percentile(idata['threshold'].to_numpy(), [2.5, 97.5])  # 90th percentile
            summary_stats['top95'].append([np.round(r, 2) for r in top95])
            full_range = np.percentile(idata['threshold'].to_numpy(), [0, 100])  # 90th percentile
            summary_stats['range'].append([np.round(r, 2) for r in full_range])
    summary_stats = pd.DataFrame(summary_stats)
    to_csv_pkl(summary_stats, stats_path, f"{isample}_descriptive_stats", _pkl=False)
