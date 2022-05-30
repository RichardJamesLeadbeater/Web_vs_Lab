import pandas as pd
from scipy import stats as st
import numpy as np
from collections import namedtuple as ntup
import os
from Code.mytools.dframe_tools import to_csv_pkl


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


og_path = os.getcwd()
ratio_path = os.path.join(og_path, 'ratios')
sum_path = os.path.join(og_path, 'summary')
sumstats_path = os.path.join(sum_path, 'stats')
stats_path = os.path.join(og_path, 'stats')
for path in [stats_path]:
    if not os.path.exists(path):
        os.makedirs(path)
measure = 'threshold'
ori = 'ori'
exp = 'task'
env = 'enviro'
p = 'observer'
expname = 'oriId_homevslab'

#
# use bootstrap data if it is available
try:
    sumdata = pd.read_csv(os.path.join(sum_path, f"{expname}_bootstrap_allsummary.csv"))
except FileNotFoundError:
    sumdata = pd.read_csv(os.path.join(sum_path, f"{expname}_allsummary.csv"))

samples = ['allweb', 'web', 'lab']
wvl_ps = pd.read_csv('subsample_ids.csv')['p_id'].to_list()
# wvl_ps = [f"p{i}" for i in range(13) if i != 0]
for isample in samples:
    summary_stats = {'sample': [], ori: [], exp: [], 'n': [], 'mean': [], 'stdev': [], 'sem': [], '95%CI': [],
                     'median': [], 'iqr': [], 'top95': [], 'range': []}
    if isample == 'allweb':
        isumdata = sumdata[sumdata[env] == 'web']
    elif isample == 'lab':
        isumdata = sumdata[sumdata[env] == 'lab']
    elif isample == 'web':
        isumdata = pd.concat([sumdata[sumdata[p] == i] for i in wvl_ps])
        isumdata = isumdata[isumdata[env] == 'web']
    if isumdata.empty:
        print(f"\n>>> DFRAME EMPTY FOR {isample}")
    for iori in sumdata[ori].unique():
        for i_task in sumdata[exp].unique():
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
    to_csv_pkl(summary_stats, sumstats_path, f"{isample}_summarystats", _pkl=False)


ratiodata = {'allweb': pd.read_csv(os.path.join(ratio_path, 'allweb', f'{expname}_allweb_allratios.csv')),
             'lab': pd.read_csv(os.path.join(ratio_path, 'lab', f'{expname}_lab_allratios.csv')),
             'web': pd.read_csv(os.path.join(ratio_path, 'web', f'{expname}_web_allratios.csv'))
             }

for key in ratiodata:
    i_expname = f"{expname}_{key}"
    i_ratiodata = ratiodata[key]
    n = len(i_ratiodata['observer'].unique())
    # test for significant differences (ratios) between experiments
    ratios_temporal = i_ratiodata[i_ratiodata[exp] == 'temporal']
    ratios_spatial = i_ratiodata[i_ratiodata[exp] == 'spatial']
    log_descriptive_stats = {exp: [], 'ratio': [], 'mean': [], 'sd': [], 'sem': []}
    lin_descriptive_stats = {exp: [], 'ratio': [], 'mean': [], 'sd': [], 'sem': []}
    log_stats_ratio = {'ratio': [], 'shapiro (p)': [], 'paired_t (t)': [], '(p)': [], 'cohens_d': []}
    lin_stats_ratio = {'ratio': [], 'shapiro (p)': [], 'paired_t (t)': [], '(p)': [], 'cohens_d': []}
    for col in i_ratiodata.columns:
        if any(i == col for i in ['OI', 'HVI', 'IOR']):  # only ratios
            # take log to put on interval scale (assumption of t-test or wilcoxon)
            lin_ratio_temporal = np.asarray(ratios_temporal[col].to_list())
            add_descriptive_to_dicty('temporal', lin_ratio_temporal, lin_descriptive_stats, col, geo_mean=False)

            lin_ratio_spatial = np.asarray(ratios_spatial[col].to_list())
            add_descriptive_to_dicty('spatial', lin_ratio_spatial, lin_descriptive_stats, col, geo_mean=False)

            perform_stats(lin_ratio_temporal, lin_ratio_spatial, lin_stats_ratio, col, 'two-sided')

            log_ratio_temporal = np.log(lin_ratio_temporal)
            add_descriptive_to_dicty('temporal', log_ratio_temporal, log_descriptive_stats, col, geo_mean=False,
                                     antilog=True)
            log_ratio_spatial = np.log(lin_ratio_spatial)
            add_descriptive_to_dicty('spatial', log_ratio_spatial, log_descriptive_stats, col, geo_mean=False,
                                     antilog=True)
            perform_stats(log_ratio_temporal, log_ratio_spatial, log_stats_ratio, col, 'two-sided')

    lin_descriptive_stats = pd.DataFrame(lin_descriptive_stats)
    lin_descriptive_stats['n'] = [n] * len(lin_descriptive_stats)
    lin_stats_ratio = pd.DataFrame.from_dict(lin_stats_ratio)
    lin_stats_ratio['n'] = [n] * len(lin_stats_ratio)
    to_csv_pkl(lin_descriptive_stats, stats_path, f'{i_expname}_descriptives_lin', _pkl=False)
    to_csv_pkl(lin_stats_ratio, stats_path, f'{i_expname}_stats_lin', _pkl=False)

    log_descriptive_stats = pd.DataFrame(log_descriptive_stats)
    log_descriptive_stats['n'] = [n] * len(log_descriptive_stats)
    log_stats_ratio = pd.DataFrame.from_dict(log_stats_ratio)
    log_stats_ratio['n'] = [n] * len(log_stats_ratio)
    to_csv_pkl(log_descriptive_stats, stats_path, f'{i_expname}_descriptives_log', _pkl=False)
    to_csv_pkl(log_stats_ratio, stats_path, f'{i_expname}_stats_log', _pkl=False)

print('')

