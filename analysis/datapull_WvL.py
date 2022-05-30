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


def get_web_raw_data(datadir):
    """ WEB RUNS """
    original_dir = os.getcwd()
    os.chdir(datadir)

    p = 'surname'
    ori = 'target orientation'
    exp = 'expName'
    level = 'ori offset'
    measure = 'proportion correct'

    # init empty DF with col names, then concatenate with list of dfs from each .csv file (pd.concat)
    column_names = [p, exp, ori, level, measure]
    web_data = []
    for filename_ in os.listdir():
        if filename_.endswith('.csv'):
            try:
                file = pd.read_csv(filename_)
                file = file[(file['current count'] == 10)]
            except:
                continue
            if file.empty:  # if run did not reach 10 reps on each ori offset then skip
                continue
            # P COMPLETED A DIFFERENT RANGE SO THIS CANNOT BE INCLUDED
            if file['surname'].unique()[0].lower() == 'leadbeater':
                if file['ori offset'].max() == 25.0:
                    continue
            file = file[column_names]  # df w/ specific cols
            naughty_list = ['test', 'testrun', 'ignore_run']  # ignore data from these names

            if any(name.lower() == file['surname'].unique()[0].lower() for name in naughty_list):
                continue  # ignore naughty list names

            # add date information (to track performance across runs)
            # hacky way of dealing with alternative naming conventions
            if filename_.split('_')[2] == 'pcn':
                file_tstamp = filename_.split('_')[4:]
            elif filename_.split('_')[2] == 'homevslab':
                file_tstamp = filename_.split('_')[3:]
            else:
                file_tstamp = filename_.split('_')[2:]
            file['date'] = f"{file_tstamp[0]}_{file_tstamp[1].split('.')[0]}"  # {date} {time}

            # add results to list
            web_data.append(file)

    web_data = pd.concat(web_data)
    web_data[p] = web_data[p].str.capitalize()
    web_data[p] = web_data[p].str.strip()
    web_data = web_data.sort_values([p, exp, ori, level, 'date'])
    web_data[level] = web_data[level].astype('str')
    web_data = add_cardinal_and_oblique(web_data, ori)
    web_data['enviro'] = ['web'] * len(web_data)
    os.chdir(original_dir)
    return web_data


def get_lab_raw_data(datadir):
    """ LAB RUNS """
    original_dir = os.getcwd()
    os.chdir(datadir)
    p = 'participant'
    ori = 'orientation'
    exp = 'task'
    level = 'offset'
    measure = 'proportion_correct'

    lab_data = []
    for p_dir in os.listdir():  # loop through dir for each p
        if p_dir == '.DS_Store':
            continue
        os.chdir(os.path.join(datadir, p_dir))
        dir_results = []
        for idx, filename in enumerate(os.listdir()):
            if filename.endswith('.psydat'):
                file = pd.read_pickle(filename)
                try:
                    file_results = file.extraInfo['analysis']['results']  # incomplete run
                except KeyError:
                    continue  # incomplete run
                n_rows = len(file_results)
                file_results[p] = [file.extraInfo[p]] * n_rows
                file_results[ori] = [file.extraInfo[ori]] * n_rows
                file_results[exp] = [file.extraInfo[exp]] * n_rows
                file_results['date'] = file.extraInfo['date']
                file_results = file_results[[p, exp, ori, level, measure, 'date']]
                dir_results.append(file_results)
        dir_results = pd.concat(dir_results)
        lab_data.append(dir_results)
    lab_data = pd.concat(lab_data)
    lab_data = add_cardinal_and_oblique(lab_data, ori)
    lab_data['enviro'] = ['lab'] * len(lab_data)
    os.chdir(original_dir)
    return lab_data


if __name__ == '__main__':

    use_bs = False
    n_resamples = 1000
    use_mp = True
    tic = time.time()

    pull_raw_data = True  # whether to pull out raw data (or use pre-loaded)

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

    subsample = ['tl', 'pvm', 'rjl', 'll', 'ski', 'lm', 'at', 'eth', 'jj', 'emh', 'ta', 'yb',
                 'fn', 'cf', 'kw', 'mws', 'mgd', 'ua', 'dm', 'vn']

    """ RAW DATA """
    raw_data = pd.read_csv(os.path.join(data_dir, 'raw_data.csv'))

    summarydata = init_cols([env, p, exp, ori, 'n_runs', 'threshold', 't_sem', 'slope', 's_sem'])

    unique = {}
    for col in raw_data.columns:
        unique[col] = raw_data[col].unique()

    # exclusions based on not reaching chance performance on obliques or cardinals
    exclusion_list = ['ayanlaja', 'power', 'sheikh', 'tasara', 'nsowah',  # pcn20
                      'ahmed', 'bird', 'dudley', 'olanubi', 'copley',  # pcn21
                      'ta', 'ua',  # WvL
                      'w', 'practice']  # misnamed

    # calculate thresholds and summarise data
    for i_env in unique[env]:
        for i_p in unique[p]:
            if any(i_p == i for i in exclusion_list):
                print(f"excluded: {i_p}")
                continue  # skip if on exclusion list
            if use_bs and any(i_p.lower() == _i for _i in subsample):
                print(f"bootstrapping: {i_p} {i_env}")
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
                    if use_bs:
                        bs_y = []  # resample data to get error of thresholds
                    n_runs = None
                    for i_lvl in x_data:
                        i_data_lvl = i_data[(i_data[lvl] == i_lvl)]
                        if i_data_lvl.empty is True:
                            continue
                        n_runs = len(i_data_lvl)
                        pcorr = i_data_lvl[dv].mean()  # mean proportion correct
                        y_data.append(pcorr)
                        if use_bs:
                            resamp_data = resample(i_data_lvl[dv], n_resamples)  # resample with replacement
                            bs_y.append(np.mean(resamp_data, 1))  # bs_mean for n_resamples
                    if not y_data:
                        continue
                    if any(str(i) == 'nan' for i in y_data):
                        print(f"{i_p}_{i_env}_{i_exp}_{i_ori}\n\tNAN DETECTED")
                        continue
                    # set 0.5 to chance level to aid curve fitting (prevent ceiling performance affecting)
                    x_data = np.asarray([0] + list(x_data))
                    y_data = np.asarray([0.5] + list(y_data))

                    datafit = fit_logistic(x_data, y_data, maxfev=4000, p0=(np.mean(x_data), 3),
                                           bounds=([0, 0], [x_data.max(), 10]))
                    if use_bs and any(i_p.lower() == _i for _i in subsample):
                        bs_y = ([np.asarray([0.5] * n_resamples)] + bs_y)  # add in 0.5 perf for fake cond (ori_diff=0)
                        bs_y = list(np.asarray(bs_y).transpose())  # transpose: each array corresponds to a single run
                        bs_thresh, bs_slope = bootstrap_logistic_fit(fit_logistic, x_data, bs_y, use_mp=use_mp)
                        # append summary_data data dicts with relevant information from this condition
                        thresh_sem = bs_thresh.sem
                        slope_sem = bs_slope.sem
                    else:
                        thresh_sem = datafit['threshold_std'] / np.sqrt(n_runs)
                        slope_sem = datafit['slope_std'] / np.sqrt(n_runs)

                    append_dicty(summarydata, [i_env, i_p, i_exp, i_ori, n_runs, datafit['threshold'],
                                               thresh_sem, datafit['slope'], slope_sem,
                                               ]
                                 )

                    print_fit = False
                    if i_p == 'p2' and i_exp == 'spatial' and any(i == i_ori for i in ['minus45']):
                        print_fit = False
                    if any(i_p == i for i in ['at']) and i_env == 'web':
                        print_fit = True
                    if print_fit:
                        plot_psychometric(x_data, y_data, datafit['x'], datafit['y'], datafit['threshold'],
                                          title=f"{i_p}\t{i_exp}\t{i_ori}", close_all=True, percent=True)
                        z = 0
    summarydata = pd.DataFrame(summarydata)

    # # replace specific values with nan or delete rows
    # failures = [('ta', 'spatial'), ('ua', 'spatial')]
    # rows = []
    # for i_p, i_task in failures:
    #     i_data = index_dframe(summarydata, columns=['observer', 'task'], values=[i_p, i_task])
    #     rows = i_data.index
    #     # summarydata.at[rows, 'threshold'] = np.nan  # change all failure task thresholds to 0 or nan
    #     summarydata = summarydata.drop(labels=rows)  # delete rows

    if use_bs:
        filename = f"{filename}_bootstrap"
    to_csv_pkl(summarydata, sum_dir, f"{filename}_allsummary", rnd=4, _pkl=False)

    # save out indexed versions for quick checking of completed conds
    labdata = summarydata[(summarydata['enviro'] == 'lab')]
    to_csv_pkl(labdata, sum_dir, f"{filename}_lab", _pkl=False)
    webdata = summarydata[(summarydata['enviro'] == 'web')]
    to_csv_pkl(webdata, sum_dir, f"{filename}_web", _pkl=False)

    if use_bs:
        print(f"{time.time() - tic}s for {n_resamples}")
