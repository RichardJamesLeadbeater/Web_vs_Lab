"""
1.  pull WEB (all and just subsample) and LAB (subsample) data
2.  logistic fit
"""

import time
import os
import numpy as np
import pandas as pd
from mytools.dframe_tools import to_csv_pkl, change_dframe_colnames, change_dframe_labels, index_dframe
from mytools.misc import add_cardinal_and_oblique
from mytools.dict_tools import append_dicty, init_cols
from mytools.mocs import plot_psychometric, fit_logistic
from mytools.bootstrap import resample, bootstrap_logistic_fit


if __name__ == '__main__':

    # common keys as vars
    env = 'enviro'
    p = 'observer'
    exp = 'task'
    ori = 'ori'
    lvl = 'offset'
    dv = 'proportion_correct'

    filename = 'oriId_homevslab'
    og_dir = os.getcwd()
    data_dir = os.path.join(og_dir, 'raw')
    sum_dir = os.path.join(og_dir, 'summary')
    sts_dir = os.path.join(sum_dir, 'stats')
    for i_dir in [data_dir, sum_dir, sts_dir]:
        if not os.path.exists(i_dir):
            os.makedirs(i_dir)

    # RAW DATA #
    raw_data = pd.read_csv(os.path.join(data_dir, 'raw_data.csv'))

    # SUMMARY DATA #
    summary_data = init_cols([env, p, exp, ori, 'n_runs', 'threshold', 't_sem', 'slope', 's_sem'])
    unique = {}
    for col in raw_data.columns:
        unique[col] = raw_data[col].unique()

    # observer label for subsample who performed web and lab conditions
    subsample = [f"p{i + 1}" for i in range(20)]  # p1 to p20
    # exclusion based on not reaching 75 % correct on a single condition
    exclusion_list = [16, 18, 22, 23, 28, 34, 39, 60, 61, 65, 66, 76, 81, 83]

    # calculate thresholds and summarise data
    for i_env in unique[env]:
        for i_p in unique[p]:
            if any(i_p == i for i in exclusion_list):
                continue  # skip if on exclusion list
            for i_exp in unique[exp]:
                for i_ori in unique[ori]:
                    i_data = raw_data[(raw_data[ori] == i_ori) &
                                      (raw_data[p] == i_p) &
                                      (raw_data[exp] == i_exp) &
                                      (raw_data[env] == i_env)]
                    if i_data.empty:
                        continue  # if cond combination doesn't exist
                    x_data = sorted(i_data[lvl].unique().astype('float'))
                    y_data = []
                    n_runs = None
                    for i_lvl in x_data:
                        i_data_lvl = i_data[(i_data[lvl] == i_lvl)]
                        if i_data_lvl.empty is True:
                            continue
                        n_runs = len(i_data_lvl)
                        pcorr = i_data_lvl[dv].mean()  # mean proportion correct
                        y_data.append(pcorr)
                    if not y_data:
                        continue
                    if any(str(i) == 'nan' for i in y_data):
                        print(f"{i_p}_{i_env}_{i_exp}_{i_ori}\n\tNAN DETECTED")
                        continue

                    # set x=0 to y=0.5 (chance) to aid curve fitting
                    x_data = np.asarray([0] + list(x_data))
                    y_data = np.asarray([0.5] + list(y_data))

                    # fit logistic curve to psychometric function
                    datafit = fit_logistic(x_data, y_data, maxfev=4000, p0=(np.mean(x_data), 3),
                                           bounds=([0, 0], [x_data.max(), 10]))

                    # standard error of the fit
                    thresh_sem = datafit['threshold_std'] / np.sqrt(n_runs)
                    slope_sem = datafit['slope_std'] / np.sqrt(n_runs)

                    # update for this condition
                    append_dicty(summary_data, [i_env, i_p, i_exp, i_ori, n_runs, datafit['threshold'],
                                                thresh_sem, datafit['slope'], slope_sem,
                                                ]
                                 )

                    # set to True to plot psychometric function and curve fit
                    print_fit = True
                    if print_fit:
                        plot_psychometric(x_data, y_data, datafit['x'], datafit['y'], datafit['threshold'],
                                          title=f"{i_p}\t{i_exp}\t{i_ori}", close_all=True, percent=True)
                        z = 0

    summary_data = pd.DataFrame(summary_data)
    to_csv_pkl(summary_data, sum_dir, f"summary_data", rnd=4, _pkl=False)
