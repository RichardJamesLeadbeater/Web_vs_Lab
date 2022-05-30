import pandas as pd
import os
from scipy import stats as st
from Code.mytools import filenav
import numpy as np
import random
import seaborn as sns
from Code.mytools.dict_tools import init_cols, append_dicty
from Code.mytools.dframe_tools import to_csv_pkl, change_dframe_labels


def get_ori_ratio(dataf, ratio, categ='orientation'):
    """calculate ratio across ? for each condition"""
    oris = None
    if ratio.upper() == 'OI':
        oris = ['oblique', 'cardinal']
    elif ratio.upper() == 'HVI':
        oris = ['vertical', 'horizontal']
    elif ratio.upper() == 'IOR':
        oris = ['plus45', 'minus45']
    x_df = dataf[dataf[categ] == oris[0]]
    x = x_df[dv].to_list()[0]
    y_df = dataf[dataf[categ] == oris[1]]
    y = y_df[dv].to_list()[0]
    ratio = x / y
    # return np.mean(ratio)
    return ratio


def get_each_ratio(dframe, conds=None, output_type=None):
    if output_type is None:
        output_type = 'table'
        cols = ratio_list
    elif output_type == 'dframe':
        cols = ['ratio score', 'value']
    if not conds:
        if output_type == 'table':
            ratio = init_cols(cols)
            ratio['OI'].append(get_ori_ratio(dframe, 'OI', categ=ori))
            # ratio['HVI'].append(get_ori_ratio(dframe, 'HVI', categ=orientation))
            # ratio['IOR'].append(get_ori_ratio(dframe, 'IOR', categ=orientation))
    else:
        if not isinstance(conds, list):
            conds = [conds]
        ratio = init_cols([c for c in conds] + cols)
        for i in dframe[conds[0]].unique():
            i_data = dframe[dframe[conds[0]] == i]
            if len(i_data) == 0:
                continue
            if len(conds) == 1:
                ratio[conds[0]].append(i)
                if output_type == 'table':
                    # ratio['HVI'].append(get_ori_ratio(i_data, 'HVI', categ=orientation))
                    # ratio['IOR'].append(get_ori_ratio(i_data, 'IOR', categ=orientation))
                    ratio['OI'].append(get_ori_ratio(i_data, 'OI', categ=ori))
                elif output_type == 'dframe':
                    # ratio['ratio score'].append('HVI')
                    # ratio['value'].append(get_ori_ratio(i_data, 'HVI', categ=orientation))
                    # ratio['ratio score'].append('IOR')
                    # ratio['value'].append(get_ori_ratio(i_data, 'IOR', categ=orientation))
                    ratio['ratio score'].append('OI')
                    ratio['value'].append(get_ori_ratio(i_data, 'OI', categ=ori))

            elif len(list(conds)) > 1:
                for j in dframe[conds[1]].unique():
                    j_data = i_data[i_data[conds[1]] == j]
                    if len(j_data) == 0:
                        continue
                    else:
                        # ratio['IOR'].append(get_ori_ratio(j_data, 'IOR', categ=ori))
                        try:
                            ratio['OI'].append(get_ori_ratio(j_data, 'OI', categ=ori))
                            ratio['HVI'].append(get_ori_ratio(j_data, 'HVI', categ=ori))
                        except IndexError:
                            pass
                        else:
                            ratio[conds[0]].append(i)
                            ratio[conds[1]].append(j)

    return pd.DataFrame.from_dict(ratio)


def mean_across_cond(dframe, condition=None):
    output = init_cols([ori, condition, dv, 'stdev', 'sem'])
    for i in dframe[ori].unique():
        i_ori = i
        if not condition:  # used when taking mean across all conds for each observer
            i_data = dframe[(dframe[ori] == i_ori)]
            append_dicty(output, [i_ori, i_data[dv].mean(), i_data[dv].std(),
                                  st.sem(i_data[dv])])
        else:
            for cond in dframe[condition].unique():  # used when taking mean across all observers for each cond
                i_data = dframe[(dframe[ori] == i_ori) & (dframe[condition] == cond)]
                append_dicty(output, [i_ori, cond, i_data[dv].mean(),
                                      i_data[dv].std(), st.sem(i_data[dv])])
    return pd.DataFrame.from_dict(output)


def get_overall_mean(ratio_data):
    all_conds_all_observers = {}
    for col in ['OI']:
        all_conds_all_observers[col] = ratio_data[col].mean()
    return pd.DataFrame(all_conds_all_observers, index=[dv])


def resample(dataset, n_samples=1000):
    dataset = np.asarray(dataset)
    # pull out resampled data for n_samples
    resampled = []
    for i in range(n_samples):
        resampled.append(random.choices(dataset, k=len(dataset)))
    return np.asarray(resampled)


def bootstrap(dframe, resample_measure):
    og_data = dframe[resample_measure]
    bs_data = resample(og_data.to_list(), n_samples=1000)

    if resample_measure == 'stdev':  # correct for bias in bs stdev
        bs_data = bs_data + (og_data.std() - np.mean(bs_data))

    return bs_data


def geomean(values):
    # put ratios on log-scale to allow us to take the mean then convert back
    return np.exp(np.mean(np.log(values)))


def get_ratio_of_means(dframe, condition, ratio_cols=None):
    if ratio_cols is None:
        ratio_cols = ['OI', 'HVI', 'IOR']
    # output = init_cols([condition, 'ratio score', 'mean_ratio'])
    output_table = init_cols([condition] + ratio_cols)
    for cond in dframe[condition].unique():  # keep separate to keep col lengths equal
        output_table[condition].append(cond)
    # calculate each oriratio for each condition
    for ratio in ratio_cols:
        for cond in dframe[condition].unique():
            i_data = dframe[(dframe[condition] == cond)]
            output_table[ratio].append(get_ori_ratio(i_data, ratio, categ=ori))
    output_table = pd.DataFrame(output_table)
    return output_table


def get_mean_of_ratios(dframe, condition, ratio_cols=None, use_geomean=False):
    if ratio_cols is None:
        ratio_cols = ['OI', 'HVI', 'IOR']
    output_table = init_cols([condition] + ratio_cols)
    for cond in dframe[condition].unique():  # keep separate to keep col lengths equal
        output_table[condition].append(cond)
    for i_ratio in ratio_cols:
        for cond in dframe[condition].unique():
            i_data = dframe[(dframe[condition] == cond)][i_ratio].to_numpy()
            if use_geomean:
                output_table[i_ratio].append(geomean(i_data))
            else:
                output_table[i_ratio].append(np.mean(i_data))
    output_table = pd.DataFrame(output_table)
    return output_table


def cols2rows(dframe, change_cols, new_col):
    # useful to convert summary_data data tables for plotting
    keep_cols = [i for i in dframe.columns if all(j != i for j in change_cols)]
    output = init_cols(keep_cols + [new_col] + ['values'])
    for i_change in change_cols:
        output[new_col] += [i_change] * len(dframe)
        output['values'] += dframe[i_change].to_list()
        for i_keep in keep_cols:
            output[i_keep] += dframe[i_keep].to_list()

    return pd.DataFrame(output)


def get_ylim_logscale(dframe, dframe_dv='mean'):
    if dframe[dframe_dv].min() > 0.1:
        minval = (dframe[dframe_dv].min() / 1.1)
        return minval, None

    # auto log scale for facetgrid can cut off smallest values, this puts ylim to suitable range
    logscale_minval = dframe[dframe_dv].min() * (10 ** dframe[dframe_dv].min())
    # leave default to auto_val (None)
    if 1 > logscale_minval >= 0.1:
        log_ymin = 0.1  # if in range of 0.1 to 1 then set min to 0.1
    elif 0.1 > logscale_minval >= 0.01:
        log_ymin = 0.01
    elif 0.01 > logscale_minval >= 0.001:
        log_ymin = 0.001
    else:
        log_ymin = None
    return log_ymin, None


def load_seaborn_prefs(style="ticks", context="talk"):
    axes_color = [0.2, 0.2, 0.2, 0.95]
    sns.set_theme(style=style, context=context,
                  rc={'axes.edgecolor': axes_color, 'xtick.color': axes_color, 'ytick.color': axes_color,
                      'axes.linewidth': 1, 'legend.title_fontsize': 0, 'legend.fontsize': 13, 'patch.linewidth': 1.2,
                      'xtick.major.width': 1, 'xtick.minor.width': 1, 'ytick.major.width': 1, 'ytick.minor.width': 1,
                      'lines.linewidth': 2.3}
                  )


def plot_each_ratio(summarytable, file_name, file_path):
    # PLOTS EACH OBSERVERS RATIO ACROSS CONDITIONS
    try:
        line_plot = sns.catplot(x=iv, y='values',
                                hue='observer', hue_order=summarytable['observer'].unique().tolist(),
                                kind='point', legend=False,
                                capsize=.04, sharey=True, palette='colorblind',
                                data=summarytable, n_boot=None, col='ratio score')
    except:
        z = 0
    all_ymin = 1000
    all_ymax = -1
    for ax in line_plot.axes:
        for subax in ax:
            i_ylim = subax.get_ylim()
            if i_ylim[0] < all_ymin:
                all_ymin = i_ylim[0]
            if i_ylim[1] > all_ymax:
                all_ymax = i_ylim[1]
            for i_collection in subax.collections:
                i_collection.set_sizes(i_collection.get_sizes() / 1.8)
    line_plot.set(ylim=(0.0, np.ceil(all_ymax)))
    # i_bar.fig.suptitle(f"Orientation ratios", size=18, y=.98)
    line_plot.set_titles(col_template="{col_name}", size=title_size, y=1, weight='bold')
    line_plot.set_xlabels(iv, size=label_size)
    line_plot.set_ylabels('ratio score', size=label_size)
    for j in line_plot.axes.flatten():
        j.tick_params(labelleft=True, labelbottom=True)
        j.set_yticklabels(j.get_yticklabels(), size=tick_size)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    line_plot.fig.subplots_adjust(wspace=width_space, hspace=height_space)
    line_plot.tight_layout()
    line_plot.fig.set(dpi=100)
    line_plot.savefig(os.path.join(file_path, f"{file_name}_each_ratio_log.png"))
    # i_bar.set(yscale='log', ylim=(all_ymin, np.ceil(all_ymax)))
    line_plot.set(yscale='log', ylim=ylimit_log)

    for j in line_plot.axes.flatten():  # removes y-axis minor ticks
        j.tick_params(labelleft=True, labelbottom=True, which='both')
        j.set_yticklabels(j.get_yticklabels(), size=tick_size, minor=True)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    line_plot.savefig(os.path.join(file_path, f"{file_name}_each_ratio_log.png"))


def plot_mean_ratios(dframe, x, y, hue, hue_order, palette, y_label='', x_label='', file_name='', file_path=''):
    bar_plot = sns.catplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                           kind='point', linestyles=[(0, (8, 0.75))] * 3, legend=False, n_boot=None)
    # change size of markers
    [i.set_sizes(i.get_sizes() / 1.8) for i in bar_plot.ax.collections]
    y_lim = bar_plot.ax.get_ylim()
    bar_plot.ax.set_ylim(0, np.ceil(y_lim[1]))
    # set labels
    bar_plot.set_xlabels(x_label, size=label_size)
    bar_plot.set_ylabels(y_label, size=label_size)

    for j in bar_plot.axes.flatten():
        j.tick_params(which='both', labelleft=True, labelbottom=True)
        j.set_yticklabels(j.get_yticklabels(), size=tick_size)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)

    bar_plot.add_legend(fontsize=legend_fontsize)
    bar_plot.tight_layout()
    bar_plot.fig.set(dpi=my_dpi)
    bar_plot.savefig(os.path.join(file_path, f"{file_name}.png"))

    bar_plot.set(yscale='log', ylim=ylimit_log)
    for j in bar_plot.axes.flatten():  # removes y-axis minor ticks
        j.tick_params(labelleft=True, labelbottom=True, which='both')
        j.set_yticklabels(j.get_yticklabels(), size=tick_size, minor=True)
        j.set_xticklabels(j.get_xticklabels(), size=tick_size)
    bar_plot.savefig(os.path.join(file_path, f"{file_path}{file_name}.png"))


def sample_data(dframe, sample):
    _sample_data = None
    if sample == 'allweb':
        _sample_data = dframe[dframe[env] == 'web']
    elif sample == 'lab':
        _sample_data = dframe[dframe[env] == 'lab']
    elif sample == 'web':
        # get p_list of subsample
        wvl_ps = pd.read_csv('subsample_ids.csv')['p_id'].to_list()
        _sample_data = pd.concat([dframe[dframe[p] == i] for i in wvl_ps])
        _sample_data = _sample_data[_sample_data[env] == 'web']
    return _sample_data


og_path = os.getcwd()
summary_path = os.path.join(og_path, 'summary')
ratios_path = os.path.join(og_path, 'ratios')
means_path = os.path.join(summary_path, 'means')
lrn_path = os.path.join(og_path, 'learning')
for i_path in [ratios_path, means_path, lrn_path]:
    if not os.path.exists(i_path):
        os.makedirs(i_path)

# get info for this experiment to use in indexing
exp_name = 'oriId_homevslab'
env = 'enviro'
iv = 'task'
p = 'observer'
ori = 'ori'
dv = 'threshold'
ratio_list = ['OI', 'HVI']

# plotting settings
load_seaborn_prefs(context='poster')
ylimit_log = (0.5, 20)
legend_fontsize = 18
label_size = 18
tick_size = 15
title_size = 19
width_space = .2
height_space = .3
my_dpi = 100

# use bootstrap data if it is available
try:
    summary_data = pd.read_csv(os.path.join(summary_path, f"oriId_homevslab_bootstrap_allsummary.csv"))
except FileNotFoundError:
    summary_data = pd.read_csv(os.path.join(summary_path, f"oriId_homevslab_allsummary.csv"))

samples = ['web', 'lab', 'allweb']
web_ratios = None
lab_ratios = None
for i_sample in samples:
    i_filename = f"{exp_name}_{i_sample}"
    i_ratio_path = os.path.join(ratios_path, i_sample)
    i_mean_path = os.path.join(means_path, i_sample)
    for i_path in [i_ratio_path, i_mean_path]:
        if not os.path.exists(i_path):
            os.makedirs(i_path)

    # todo fix issue when i_sample == 'web
    i_summary = sample_data(summary_data, i_sample)

    # useful to compare ratios that change as a function of the IV, and across participants
    each_ratio = get_each_ratio(i_summary, [p, iv])
    to_csv_pkl(each_ratio, i_ratio_path, f"{i_filename}_allratios", _pkl=False)
    if i_sample == 'web':
        web_ratios = each_ratio
    if i_sample == 'lab':
        lab_ratios = each_ratio

    # mean threshold score in each condition
    mean_of_conds = mean_across_cond(summary_data, iv)
    to_csv_pkl(mean_of_conds, i_mean_path, f"{i_filename}_iv", rnd=6, _pkl=False)

    # ratio of mean threshold scores
    ratio_of_means = get_ratio_of_means(mean_of_conds, iv, ratio_cols=ratio_list)
    to_csv_pkl(ratio_of_means, i_ratio_path, f"{i_filename}_iv_ratioofmeans", _pkl=False)

    # arithmetic mean of the individual ratios
    mean_of_ratios = get_mean_of_ratios(each_ratio, iv, ratio_cols=ratio_list, use_geomean=False)
    to_csv_pkl(mean_of_ratios, i_ratio_path, f"{i_filename}_iv_meanofratios", _pkl=False)
    # geometric mean of the individual ratios
    geomean_of_ratios = get_mean_of_ratios(each_ratio, iv, ratio_cols=ratio_list, use_geomean=True)
    to_csv_pkl(geomean_of_ratios, i_ratio_path, f"{i_filename}_iv_geomeanofratios", _pkl=False)

    # # # Plot Ratios # # #
    # plot each observer's ratios on each condition
    plot_data = cols2rows(each_ratio, ratio_list, 'ratio score')  # plot friendly dframe
    plot_each_ratio(plot_data, file_name=i_filename, file_path=i_ratio_path)

    # ratio of mean thresholds
    plot_data = cols2rows(ratio_of_means, ratio_list, 'ratio score')  # plot friendly dframe
    plot_mean_ratios(plot_data, x=iv, y='values', hue='ratio score', hue_order=ratio_list, palette='colorblind',
                     y_label='mean ratio score', x_label=iv, file_name=i_filename, file_path=i_ratio_path)

    # arithmetic mean of individual ratios
    plot_data = cols2rows(mean_of_ratios, ratio_list, 'ratio score')  # plot friendly dframe
    plot_mean_ratios(plot_data, x=iv, y='values', hue='ratio score', hue_order=ratio_list, palette='colorblind',
                     y_label='mean ratio score', x_label=iv, file_name=i_filename, file_path=i_ratio_path)

    # geometric mean of individual ratios
    plot_data = cols2rows(geomean_of_ratios, ratio_list, 'ratio score')  # plot friendly dframe
    plot_mean_ratios(plot_data, x=iv, y='values', hue='ratio score', hue_order=ratio_list, palette='colorblind',
                     y_label='geomean ratio score', x_label=iv, file_name=i_filename, file_path=i_ratio_path)

# todo compare OI magnitude of web vs lab (line) in each task (subplots) for each observer (separate graphs)
web_ratios['enviro'] = ['web'] * len(web_ratios)
lab_ratios['enviro'] = ['lab'] * len(lab_ratios)
wvl_ratios = pd.concat([web_ratios, lab_ratios])
wvl_ratio_diff = np.asarray(web_ratios['OI'] - lab_ratios['OI'])
z = 0

# todo compare OEI between 1st and 2nd exp
#      x-axis=1st_vs_2nd
wvl_ps_1st = pd.read_csv('subsample_ids.csv', index_col=0).to_dict()['first']

# split summary data between 1st vs 2nd
data_1st = []
data_2nd = []
for i_p in wvl_ps_1st:
    i_data = summary_data[(summary_data[p] == i_p)]

    lbl_1 = wvl_ps_1st[i_p]
    i_data_1 = i_data[(i_data[env] == lbl_1)]
    i_data_2 = i_data[(i_data[env] != lbl_1)]  # only works if there are exactly 2 enviros

    data_1st.append(i_data_1)
    data_2nd.append(i_data_2)
data_1st = pd.concat(data_1st)
data_2nd = pd.concat(data_2nd)

data_1st['session'] = 1
data_2nd['session'] = 2

wvl_data = pd.concat([data_1st, data_2nd])

# mean threshold score for each ori across exp sessions
means_lrn = []
for i_task in wvl_data[iv].unique():
    i_data = wvl_data[(wvl_data[iv] == i_task)]
    i_means = mean_across_cond(i_data, 'session')
    i_means[iv] = i_task  # create relevant col
    means_lrn.append(i_means)
means_lrn = pd.concat(means_lrn)
means_lrn = means_lrn[[iv, ori, 'session', dv, 'stdev', 'sem']]
to_csv_pkl(means_lrn, lrn_path, f"mean_thresholds", rnd=2, _pkl=False)

# ratio of mean threshold scores
ratios_lrn = []
for i_task in means_lrn[iv].unique():
    i_data = means_lrn[(means_lrn[iv] == i_task)]
    i_ratios = get_ratio_of_means(i_data, 'session', ratio_cols=ratio_list)
    i_ratios[iv] = i_task
    ratios_lrn.append(i_ratios)
ratios_lrn = pd.concat(ratios_lrn)
ratios_lrn = ratios_lrn[[iv, 'session', 'OI', 'HVI']]
to_csv_pkl(ratios_lrn, lrn_path, f"ratios_of_mean_thresholds", rnd=2, _pkl=False)

z=0

# todo perform other actions on 1st vs 2nd data
# get each ratio of each observer in each session
filename = "1st_vs_2nd"
each_ratio_lrn = []
for i_task in wvl_data[iv].unique():
    i_data = wvl_data[(wvl_data[iv] == i_task)]
    i_each = get_each_ratio(i_data, [p, 'session'])
    i_each[iv] = i_task
    each_ratio_lrn.append(i_each)
each_ratio_lrn = pd.concat(each_ratio_lrn)
to_csv_pkl(each_ratio_lrn, lrn_path, f"{filename}_each_ratio", rnd=2, _pkl=False)

arimeans = []
geomeans = []
for i_task in each_ratio_lrn[iv].unique():
    i_data = each_ratio_lrn[(each_ratio_lrn[iv] == i_task)]
    i_ari = get_mean_of_ratios(i_data, 'session', ratio_cols=ratio_list, use_geomean=False)
    i_geo = get_mean_of_ratios(i_data, 'session', ratio_cols=ratio_list, use_geomean=True)
    i_geo[iv] = i_task
    arimeans.append(i_ari)
    geomeans.append(i_geo)
arimeans = pd.concat(arimeans)
geomeans = pd.concat(geomeans)
to_csv_pkl(arimeans, lrn_path, f"{filename}_arimean_of_ratios", rnd=2, _pkl=False)
to_csv_pkl(geomeans, lrn_path, f"{filename}_geomean_of_ratios", rnd=2, _pkl=False)

mean_of_ratios = get_mean_of_ratios(each_ratio, iv, ratio_cols=ratio_list, use_geomean=False)
to_csv_pkl(mean_of_ratios, lrn_path, f"{filename}_iv_meanofratios", _pkl=False)
# geometric mean of the individual ratios
geomean_of_ratios = get_mean_of_ratios(each_ratio, iv, ratio_cols=ratio_list, use_geomean=True)
to_csv_pkl(mean_of_ratios, lrn_path, f"{filename}_iv_geomeanofratios", _pkl=False)

