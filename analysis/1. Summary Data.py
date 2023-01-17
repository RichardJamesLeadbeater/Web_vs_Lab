import os
import numpy as np
import pandas as pd
from mytools.dframe_tools import index_dframe, to_csv_pkl
from mytools.dict_tools import append_dicty, init_cols
from mytools.mocs import plot_psychometric, fit_logistic

# # # get summary_data by calculating threshold for each condition # # #

# common keys as vars
env = 'enviro'
p = 'observer'
exp = 'task'
ori = 'ori'
lvl = 'offset'
dv = 'proportion_correct'

og_dir = os.getcwd()
data_dir = os.path.join(og_dir, 'data')
raw_dir = os.path.join(data_dir, 'raw')
sum_dir = os.path.join(data_dir, 'summary')
for i_dir in [sum_dir]:
    if not os.path.exists(i_dir):
        os.makedirs(i_dir)

# RAW DATA #
raw_data = pd.read_csv(os.path.join(raw_dir, 'raw_data.csv'))
raw_data = index_dframe(raw_data, ['ori'], ['horizontal', 'vertical', 'minus45', 'plus45'], 'or')  # only these oris

# SUMMARY DATA #
summary_data = init_cols([env, p, exp, ori, 'n_runs', 'threshold', 't_sem', 'slope', 's_sem'])
unique = {}
for col in raw_data.columns:
    unique[col] = raw_data[col].unique()

# calculate thresholds for each observer on each condition (fit curve to psychometric function)
for i_env in unique[env]:
    for i_p in unique[p]:
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
                                            thresh_sem, datafit['slope'], slope_sem])

                print_fit = False  # to plot curve fit on psychometric function
                if print_fit:
                    plot_psychometric(x_data, y_data, datafit['x'], datafit['y'], datafit['threshold'],
                                      title=f"{i_p}\t{i_exp}\t{i_ori}", close_all=True, percent=True)

summary_data = pd.DataFrame(summary_data)
to_csv_pkl(summary_data, sum_dir, f"summary_data", rnd=4, _pkl=False)
