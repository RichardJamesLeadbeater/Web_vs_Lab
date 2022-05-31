import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from scipy import stats as st
from mytools.misc import add_cardinal_and_oblique
from mytools.dict_tools import append_dicty
from mytools.dframe_tools import to_csv_pkl, change_dframe_colnames, change_dframe_labels, index_dframe
from mytools.my_seaborn import load_seaborn_prefs, barplot_err


def rand_jitter(arr, multiplier=1.04):
    arr = [i for i in np.asarray(arr)]
    for idx, i in enumerate(arr):
        arr[idx] = np.random.choice([i * multiplier, i * (multiplier / 1.02), i * (multiplier * 1.02),
                                     i / multiplier, i / (multiplier / 1.02), i / (multiplier * 1.02)]
                                    )
    return arr


def run_icc(dframe_sum, subjects, raters, scores, cond1, cond2=None, icctype='ICC3'):
    icc_analyses = {}
    icc_summary = {}
    icc_cols = ['ICC', 'F', 'df1', 'df2', 'pval', 'CI95%']
    titlelabel = 'title'
    for icol in (['Type'] + [titlelabel] + icc_cols):
        icc_summary[icol] = []

    for i_cond1 in dframe_sum[cond1].unique():
        i_dframe1 = dframe_sum[(dframe_sum[cond1] == i_cond1)]
        # whole icc analysis
        icc_1 = pg.intraclass_corr(i_dframe1, subjects, raters, scores, nan_policy='omit')
        icc_analyses[i_cond1] = icc_1.copy()
        # pull out ICC of interest e.g. ICC3
        icc_row1 = icc_1[(icc_1['Type'] == icctype)]
        for icol in icc_cols:
            icc_summary[icol].append(icc_row1[icol].to_list()[0])
        icc_summary[titlelabel].append(i_cond1)
        icc_summary['Type'].append(icctype)

    if not cond2:
        icc_summary = pd.DataFrame(icc_summary)
        return icc_summary, icc_analyses

    for i_cond2 in dframe_sum[cond2].unique():
        # analyse indiv conds
        i_dframe2 = dframe_sum[(dframe_sum[cond2] == i_cond2)]
        # whole icc analysis
        icc_2 = pg.intraclass_corr(i_dframe2, subjects, raters, scores, nan_policy='omit')
        icc_analyses[i_cond2] = icc_2.copy()
        # pull out ICC of interest e.g. ICC3
        icc_row2 = icc_2[(icc_2['Type'] == icctype)]
        for icol in icc_cols:
            icc_summary[icol].append(icc_row2[icol].to_list()[0])
        icc_summary[titlelabel].append(i_cond2)
        icc_summary['Type'].append(icctype)

    for i_cond1 in dframe_sum[cond1].unique():
        for i_cond2 in dframe_sum[cond2].unique():
            # analyse combined conds
            i_dframe_mix = dframe_sum[(dframe_sum[cond1] == i_cond1) &
                                      (dframe_sum[cond2] == i_cond2)]
            icc_mix = pg.intraclass_corr(i_dframe_mix, subjects, raters, scores, nan_policy='omit')
            icc_analyses[f"{i_cond1}_{i_cond2}"] = icc_mix.copy()
            # pull out ICC of interest e.g. ICC3
            icc_rowmix = icc_mix[(icc_mix['Type'] == icctype)]
            for icol in icc_cols:
                icc_summary[icol].append(icc_rowmix[icol].to_list()[0])
            icc_summary[titlelabel].append(f"{i_cond1}_{i_cond2}")
            icc_summary['Type'].append(icctype)

    icc_summary = pd.DataFrame(icc_summary)
    return icc_summary, icc_analyses


def geomean(values):
    # put ratios on log-scale to allow us to take the mean then convert back
    return np.exp(np.mean(np.log(values)))


def plot_bland_altman(dframe, x, y, meandiff, loa, loe, title, xlims, ylims, hue, palette, markers, style,
                      fig=None):
    if fig is None:
        plt.figure()

    scatplot = sns.scatterplot(data=dframe, x=x, y=y, hue=hue, palette=palette, alpha=0.6,
                               legend='full', markers=['P', '^', 'd', 'X'], style='ori', s=markersize)
    leg = scatplot.get_legend()
    try:
        scatplot.get_legend().remove()  # necessary step so 'full' will work
        # scatplot.legend(loc='upper center', ncol=2, framealpha=.7, markerscale=0.6, borderpad=.5,
        #                 columnspacing=.7, handletextpad=-.4, scatteryoffsets=[0.45])
        # # remove titles from legend
        # handles, labels = scatplot.get_legend_handles_labels()
        # for i, idx, in enumerate(handles):
        #     if any(handles[idx].get_label() == j for j in ['task', 'ori']):
        #         handles[idx] = ''
        #         labels[idx] = ''
        # scatplot.legend(handles=handles, labels=labels)
    except AttributeError:
        z = 0
        pass
    scatplot.axes.plot(xlims, [meandiff, meandiff], c='r', ls='--', linewidth=1.5)
    scatplot.axes.plot(xlims, [loa[0], loa[0]], c='b', ls=':', linewidth=2)
    scatplot.axes.plot(xlims, [loa[1], loa[1]], c='b', ls=':', linewidth=2)
    scatplot.axes.plot(xlims, [loe, loe], c=[0, 0, 0, 0.2], ls='-', linewidth=2)
    # scatplot.set_title(title, y=1.03)
    scatplot.set(xlim=xlims, ylim=ylims)
    scatplot.figure.tight_layout()
    return scatplot


def bland_altman(dframe, title, labels, loglabels, ratiolabel, hue=None,
                 show_linear=False, show_log=False, show_ratio=False, save=False, style=None, markerlist=None,
                 checkifnormal=False, bias_adjust=False, save_dir=''):
    # _l_meandiff = line of mean-difference
    # _l_agree = lines of agreement
    # _l_equal = line of equality
    if title == 'Web vs Lab':
        z = 0
    # labels correspond to [diff, mean]
    diff = labels[0]
    ave = labels[1]
    # labels correspond to [diff_log()), mean_log()]
    logdiff = loglabels[0]
    logave = loglabels[1]
    # color_palette = sns.color_palette('colorblind', 6)[4:6]
    if len(dframe[hue].unique()) == 2:
        palette = [manual_color[1]] + [manual_color[4]]
    if len(dframe[hue].unique()) == 4:
        palette = sns.color_palette(manual_color)

    if checkifnormal:
        # check if differences of log-transformed data are normally distributed
        check_normal_dist(dframe[logdiff], f"{title} log", show_hist=True)
        if title == 'Web vs Lab':
            z = 0
        print(f"log: {title}\n")
        plt.close('all')
        # check_normal_dist(dframe[diff], f"{title} lin", show_hist=True)
        # print(f"lin: {title}\n")

    # LOG TRANSFORMED BLAND-ALTMAN
    # set lower and upper limit of agreement

    # for linear plots
    lin_xlims = [0, 28]  # linear scale
    lin_ylims = [None, None]
    lin_l_meandiff = dframe[diff].mean()  # line of mean difference
    lin_l_agree = [lin_l_meandiff + (1.96 * dframe[diff].std(ddof=1)) * i for i in [-1, 1]]
    if bias_adjust:
        lin_l_agree -= lin_l_meandiff  # shifts loa around meandiff
    lin_l_equal = 0

    # for log plots
    log_xlims = [-0.2, np.log(lin_xlims[1])]
    log_ylims = [-1.2, 1.2]  # ensures on same scale
    log_l_meandiff = dframe[logdiff].mean()
    log_l_agree = [log_l_meandiff + (1.96 * dframe[logdiff].std()) * i for i in [-1, 1]]
    if bias_adjust:
        log_l_agree -= log_l_meandiff
    log_l_equal = 0
    log_se = st.sem(dframe[logdiff])

    # for ratio plots
    rat_ylims = [0.44, 2.2]  # ratio scale extend out a little for nicer plots
    log_ylims = [np.log(i) for i in rat_ylims]
    ratio_l_meandiff = geomean(dframe[ratiolabel])  # calc geometric mean of all lab/web ratios
    ratio_l_equal = 1
    # difference between two logs is equal to the ratio
    ratio_l_agree = [np.exp(i) for i in log_l_agree]
    # print(f"{title}:\tag={ratio_l_agree}\tmd={rat_l_meandiff}")
    if show_log and show_ratio:
        # all steps below ensure values on each y-axes line up
        plotdframe = dframe.copy()
        plotdframe[logave] = rand_jitter(plotdframe[logave], 1.03)  # jitter x-axis for shared datapoints
        baplot = plot_bland_altman(plotdframe, logave, logdiff, log_l_meandiff, log_l_agree, log_l_equal, title,
                                   log_xlims, log_ylims, hue, palette, markerlist, style)
        baplot.set_ylabel(logdiff, fontweight='bold', labelpad=10, size=18)
        baplot.set_xlabel(logave, fontweight='bold', labelpad=8, size=18)
        # add second y-axis with ratios
        ax2 = baplot.twinx()
        baplot = plot_bland_altman(plotdframe, logave, ratiolabel, ratio_l_meandiff, ratio_l_agree, ratio_l_equal, title,
                                   log_xlims, rat_ylims, hue, palette, markerlist, style,
                                   fig=ax2)
        ax2.set_ylabel(ratiolabel, fontweight='bold', labelpad=10, size=18)
        ax2.set(yscale='log', ylim=rat_ylims, xlim=log_xlims)  # ratios on log scale
        # # show actual ratio
        # ax2.set_yticks([(1 / 1.75), (1 / 1.25), 1, 1.25, 1.75], minor=True)  # remove minor ticks
        # ax2.set_yticklabels(["1 : 2", "1 : 1.5", "1 : 1", "1.5 : 1", "2 : 1"] * 5, minor=True)  # remove minor ticks
        # ax2.set_yticks([0.5, (1 / 1.5), 1, 1.5, 2], minor=False)  # set values then labels of minor ticks
        # ax2.set_yticklabels(["1 : 2", "1 : 1.5", "1 : 1", "1.5 : 1", "2 : 1"], minor=False, size=14, fontweight="bold")
        # show axes of interest
        ax2.set_yticks(sorted(np.hstack([1 / np.linspace(1.1, 2, 10), np.linspace(1.1, 2, 10)])), minor=True)  # remove minor ticks
        ax2.set_yticklabels([""] * len(ax2.get_yticks(minor=True)), minor=True)
        ax2.set_yticks([0.5, (1 / 1.5), 1, 1.5, 2], minor=False)  # set values then labels of minor ticks
        ax2.set_yticklabels([f"{y:.2f}" for y in ax2.get_yticks()], minor=False, size=15)
        # ax2.set_yticks([1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2], minor=False)  # set values then labels of minor ticks
        # ax2.set_yticklabels(['0.50', '0.66', '0.80', '1.00', '1.25', '1.50', '2.00'], minor=False, size=13)
        ax2.set_alpha(0)
        if save:
            baplot.figure.set(dpi=mydpi)
            if bias_adjust:
                baplot.figure.savefig(os.path.join(save_dir, f"BA_logdiff_ratio_{title}_biasadjust.png"))
            else:
                baplot.figure.savefig(os.path.join(save_dir, f"BA_logdiff_ratio_{title}.png"))
        plt.close('all')
    # BA PLOT ON LOG-TRANSFORMED DATA
    if show_log and not show_ratio:
        # the below steps ensures both values on both y-axes line up
        baplot = plot_bland_altman(dframe, ave, logdiff, log_l_meandiff, log_l_agree, log_l_equal, title,
                                   log_xlims, log_ylims, hue, palette, markerlist, style)

    # RATIO BLAND-ALTMAN
    # taking the antilog gives the limits of agreement for the ratio between the two measures (Bland & Altman, 1999)
    if show_ratio and not show_log:
        baplot = plot_bland_altman(dframe, ave, ratiolabel, ratio_l_meandiff, ratio_l_agree, ratio_l_equal, title,
                                   lin_xlims, rat_ylims, hue, palette, markerlist, style)
        if save:
            baplot.figure.set(dpi=mydpi)
            baplot.figure.savefig(os.path.join(save_dir, f"BA_ratio_{title}.png"))

    # LINEAR BLAND-ALTMAN (not accurate if variance proportional with threshold magnitude)
    if show_linear:
        baplot = plot_bland_altman(dframe, ave, diff, lin_l_meandiff, lin_l_agree, lin_l_equal, title,
                                   lin_xlims, lin_ylims, hue, palette, markerlist, style)
        if save:
            baplot.figure.set(dpi=mydpi)
            baplot.figure.savefig(os.path.join(save_dir, f"BA_lindiff_{title}.png"))

    return {'mean_diff': ratio_l_meandiff, 'loa': ratio_l_agree}


def check_normal_dist(data, title=None, nbins=13, show_hist=False):
    data = np.asarray(data)
    if show_hist:
        plt.figure()
        sns.histplot(data, bins=nbins)
        plt.title(title)
        plt.tight_layout()
    shapwilk = st.shapiro(data)
    print(f"p = {shapwilk[1]:.3f}")
    if shapwilk[1] < 0.05:
        isnormal = False
        print('significantly deviates from normal distribution')
    else:
        isnormal = True
    return isnormal


def get_median_absolute_ratio(yy_, logdiff_col, title=''):
    check_normal_dist(yy_[logdiff_col], title='absdiffs', show_hist=False)
    absdiffs = np.asarray(abs(yy_[logdiff_col]))
    abs_gmean = np.exp(np.mean(absdiffs))
    abs_median = np.exp(np.median(absdiffs))  # median of absolute diffs as does not follow normal dist
    n_ = len(absdiffs)
    absgse = np.exp(np.std(absdiffs, ddof=1) / np.sqrt(n_))
    iqr = np.exp(np.percentile(absdiffs, [25, 75]))
    print(f"{title}:\n"
          f"\tabsGM = {abs_gmean}\n"
          f"\tabsGSE = {absgse}\n"
          f"\tmedian_abs_ratio = {abs_median} {iqr[1] - iqr[0]}\n")
    return abs_median, np.round(iqr, 2)


# init paths #
og_dir = os.getcwd()
data_dir = os.path.join(og_dir, 'data')
raw_dir = os.path.join(data_dir, 'raw')
sum_dir = os.path.join(data_dir, 'summary')
# new dirs
rel_dir = os.path.join(og_dir, 'reliability_analyses')
sub_dir = os.path.join(rel_dir, 'yy_data')
gph_dir = os.path.join(rel_dir, 'graphs')
sts_dir = os.path.join(rel_dir, 'stats')

for i_dir in [sub_dir, sts_dir, gph_dir]:
    if not os.path.exists(i_dir):
        os.makedirs(i_dir)

# common keys as vars
env = 'enviro'
p = 'observer'
exp = 'task'
ori = 'ori'
lvl = 'offset'
dv = 'proportion_correct'
mydpi = 100  # plot setting

# load in summary data #
summary_data = pd.read_csv(os.path.join(sum_dir, f"summary_data.csv"))

"""
Intraclass Correlation
"""
# only use data in icc that has runs in web and lab
icc_datalist = []
for i_p in summary_data[p].unique():
    for i_exp in summary_data[exp].unique():
        for i_ori in summary_data[ori].unique():
            i_data = summary_data[(summary_data[p] == i_p) &
                                  (summary_data[exp] == i_exp) &
                                  (summary_data[ori] == i_ori)]
            if len(i_data[env].unique()) < 2:
                continue  # skip if only performed web
            icc_datalist.append(i_data)
icc_data = pd.concat([i for i in icc_datalist])
icc, _ = run_icc(icc_data, p, env, 'threshold', cond1=exp, cond2=ori, icctype='ICC2')
to_csv_pkl(icc, sts_dir, f"icc_analysis_{icc['Type'].to_list()[0]}", rnd=3, _pkl=False)

""" YY DATA """
# manipulate data ready for correlation / yyplot / bland-altman analyses
# set plotting preferences
load_seaborn_prefs(context='talk')

# format dataframe to work with sns.regplot()... e.g., YY plot with regression line
lab_summary = summary_data[(summary_data[env] == 'lab')]
web_summary = summary_data[(summary_data[env] == 'web')]

# put lab_threshold and web_threshold as independent columns for use in sns plots
yy_data = {}
cols = [p, exp, ori, 'lab', 'web', 'lab_nruns', 'web_nruns']
for col in cols:
    yy_data[col] = []
for i_p in summary_data[p].unique():
    for i_exp in summary_data[exp].unique():
        for i_ori in summary_data[ori].unique():
            i_lab = lab_summary[(lab_summary[p] == i_p) &
                                (lab_summary[exp] == i_exp) &
                                (lab_summary[ori] == i_ori)]
            i_web = web_summary[(web_summary[p] == i_p) &
                                (web_summary[exp] == i_exp) &
                                (web_summary[ori] == i_ori)]
            if any(i.empty for i in [i_lab, i_web]):
                continue  # data doesn't exist for both lab and web
            lab_thresh = i_lab['threshold'].to_list()[0]
            web_thresh = i_web['threshold'].to_list()[0]
            append_dicty(yy_data, [i_p, i_exp, i_ori, i_lab['threshold'].to_list()[0], i_web['threshold'].to_list()[0],
                                   i_lab['n_runs'].to_list()[0], i_web['n_runs'].to_list()[0]])
            print('')
yy_data = pd.DataFrame(yy_data)
# add information for Bland-Altman plots
diff_label = 'lab - web'
mean_label = '(lab + web) / 2'
yy_data[diff_label] = yy_data['lab'] - yy_data['web']
yy_data[mean_label] = np.mean([yy_data['lab'], yy_data['web']], 0)
# log transformed data to deal with with increased variance with increased magnitude
logdiff_label = 'log(lab) - log(web)'
logmean_label = '(log[lab] + log[web]) / 2 '
yy_data[logdiff_label] = np.log(yy_data['lab']) - np.log(yy_data['web'])
yy_data[logmean_label] = np.mean([np.log(yy_data['lab']), np.log(yy_data['web'])], 0)
ratio_label = 'threshold ratio (lab / web)'
yy_data[ratio_label] = np.asarray(yy_data['lab']) / np.asarray(yy_data['web'])
to_csv_pkl(yy_data, sub_dir, 'subsample_data', rnd=3, _pkl=False)

# # # ANALYSE THE ABSOLUTE MAGNITUDE RATIO VALUES # # #
abs_log_diff = abs(yy_data[logdiff_label])
abs_ratio_vals = np.exp(abs_log_diff)
median_abs_ratio = np.median(abs_ratio_vals)
percentiles = {}
for i_pcnt in [98.75, 97.5, 95, 90, 75, 60, 50]:
    percentiles[str(i_pcnt)] = np.percentile(abs_ratio_vals, i_pcnt)

threshold_ratios = {'task': [], 'ori': [], 'median_abs_ratio': [], '95th_percentile': []}
for i_task in ['temporal', 'spatial']:
    for i_ori in ['horizontal', 'vertical', 'minus45', 'plus45']:
        i_data = yy_data[(yy_data['ori'] == i_ori) & (yy_data['task'] == i_task)]
        abs_log_diff = abs(i_data[logdiff_label])
        abs_ratio_vals = np.exp(abs_log_diff)
        median_abs_ratio = np.median(abs_ratio_vals)
        percentile = np.percentile(abs_ratio_vals, 95)
        append_dicty(threshold_ratios, [i_task, i_ori, median_abs_ratio, percentile])
threshold_ratios = pd.DataFrame(threshold_ratios)
yylim = summary_data['threshold'].max()

"""
YY Plot
"""
color_palette = sns.color_palette('colorblind', 4)
color_palette = [[0.2]*3, [0.8]*3, color_palette[2], color_palette[3]]
manual_color = [i for i in color_palette]
markersize = 150
jitter_size = 1.03
figdims = (4, 4)
for itask in ['spatial', 'temporal']:
    ifig, iax = plt.subplots(figsize=figdims)
    iplotdata = yy_data[yy_data[exp] == itask]
    iplotdata['web'] = rand_jitter(iplotdata['web'], 1.03)
    iplotdata['lab'] = rand_jitter(iplotdata['lab'], 1.02)
    corr_plot = sns.scatterplot(x='web', y='lab', data=iplotdata, hue=ori, ci=None, alpha=1,
                                palette=color_palette, markers=['P', '^', 'd', 'X'], style='ori',
                                s=markersize)
    corr_plot.plot([0, 27.5], [0, 27.5], c=[0, 0, 0, 0.3], ls='--', linewidth=3)
    corr_plot.set_title(corr_plot.get_title().split('= ')[-1])
    corr_plot.set_ylabel('lab threshold ($^\circ$)', fontweight='bold')
    corr_plot.set_xlabel('web threshold ($^\circ$)', fontweight='bold')
    corr_plot.set(xlim=[-2, yylim + 2], ylim=[-2, yylim + 2])
    xymajorticks = np.linspace(0, 25, 6)
    corr_plot.set_yticks(xymajorticks)
    corr_plot.set_xticks(xymajorticks)
    corr_plot.set_yticklabels([f"{int(i)}" for i in xymajorticks])
    corr_plot.set_xticklabels([f"{int(i)}" for i in xymajorticks])
    xyminorticks = np.linspace(2.5, 27.5, 6)
    corr_plot.set_yticks(xyminorticks, minor=True)
    corr_plot.set_xticks(xyminorticks, minor=True)
    corr_plot.spines['top'].set_visible(False)
    corr_plot.spines['right'].set_visible(False)
    corr_plot.get_legend().remove()
    corr_plot.figure.tight_layout()
    corr_plot.figure.savefig(os.path.join(gph_dir, f"correlation_{itask}.png"))
    plt.close()
# slope, intercept, r_value, p_value, std_err


"""
Bland-Altman Plot
"""

# y-axis: difference between two measures
# x-axis: average of two measures
# red line: mean difference
# blue lines: limits of agreement
plt.close('all')
# separate Bland-Altman plots for each task (linear means diffs better-follow norm_dist)
yydata_spatial = yy_data[(yy_data['task'] == 'spatial')]
yydata_temporal = yy_data[(yy_data['task'] == 'temporal')]
ratiostats = {'task': [], 'geomean_ratio': [], 'loa': [], 'median_ratio': [], 'iqr': [],
              'median_abs_ratio': [], 'iqr_abs': [], 'q1': [], 'q3': []}
for yy in [(yydata_spatial, 'spatial'), (yydata_temporal, 'temporal')]:
    i_data = yy[0]
    i_title = yy[1]
    ratiostats[exp].append(i_data[exp].to_list()[0])
    ba_data = bland_altman(i_data, i_title, hue=ori, labels=[diff_label, mean_label],
                           loglabels=[logdiff_label, logmean_label], ratiolabel=ratio_label,
                           save=True, show_ratio=True, show_log=True, show_linear=False, markerlist=['X'],
                           bias_adjust=False, checkifnormal=True, save_dir=gph_dir)
    ratiostats['geomean_ratio'].append(ba_data['mean_diff'])
    ratiostats['loa'].append(np.round(ba_data['loa'], 2))
    ratiostats['median_ratio'].append(np.median(i_data[ratio_label]))
    quartiles = np.round(np.percentile(i_data[ratio_label], [25, 75]), 2)
    ratiostats['iqr'].append(quartiles[1] - quartiles[0])
    abs_ratio_stats = get_median_absolute_ratio(i_data, logdiff_label, i_title)
    ratiostats['median_abs_ratio'].append(abs_ratio_stats[0])
    ratiostats['iqr_abs'].append(abs_ratio_stats[1][1] - abs_ratio_stats[1][0])
    ratiostats['q1'].append(abs_ratio_stats[1][0])
    ratiostats['q3'].append(abs_ratio_stats[1][1])
ratiostats = pd.DataFrame(ratiostats)
to_csv_pkl(ratiostats, sts_dir, 'blandaltman_stats', rnd=2, _pkl=False)

# median abs ratio for each ori / task
median_data = {'ori': [], 'task': [], 'median_abs_ratio': [], 'iqr': [], 'q1': [], 'q3': []}
for i_ori in yy_data[ori].unique():
    for i_task in yy_data[exp].unique():
        i_data = index_dframe(yy_data, [ori, exp], [i_ori, i_task])
        median_data[ori].append(i_ori)
        median_data[exp].append(i_task)
        abs_ratio_stats = get_median_absolute_ratio(i_data, logdiff_label, f"{i_ori}_{i_task}")
        median_data['median_abs_ratio'].append(abs_ratio_stats[0])
        median_data['iqr'].append(abs_ratio_stats[1][1] - abs_ratio_stats[1][0])
        median_data['q1'].append(abs_ratio_stats[1][0])
        median_data['q3'].append(abs_ratio_stats[1][1])
median_data = pd.DataFrame(median_data)
to_csv_pkl(median_data, sts_dir, 'abs_median_ratio', rnd=2, _pkl=False)

# check for violations of limits of agreement
violations = {'id': [], 'violation': []}
for i_p in yy_data[p].unique():
    for i_task in yy_data[exp].unique():
        for i_ori in yy_data[ori].unique():
            i_data = index_dframe(yy_data, [p, exp, ori], [i_p, i_task, i_ori])
            wvl_ratio = float(i_data[ratio_label])
            loa = list(ratiostats[(ratiostats[exp] == i_task)]['loa'])[0]
            if wvl_ratio < loa[0]:
                # print(f"{i_p}_{i_task}_{i_ori} violates lower loa")
                violations['id'].append(f"{i_p}_{i_task}_{i_ori}")
                violations['violation'].append('lower')
            elif wvl_ratio > loa[1]:
                violations['id'].append(f"{i_p}_{i_task}_{i_ori}")
                violations['violation'].append('upper')
                # print(f"{i_p}_{i_task}_{i_ori} violates upper loa")
to_csv_pkl(pd.DataFrame(violations), sts_dir, 'blandaltman_loa_violations', _pkl=False)


"""
Learning Effect
"""
# # # check for whether participants improved between 1st and 2nd session # # #

# label and which enviro first for subsample
p_info = pd.read_csv(os.path.join(og_dir, 'subsample_ids.csv'))

web_1st_id = list(p_info[p_info['first'] == 'web']['p_id'])
web_1st_data = pd.concat([yy_data[yy_data[p] == i.lower()] for i in web_1st_id])
web_1st_data = change_dframe_colnames(web_1st_data, ['web', 'lab'], ['1st', '2nd'])

lab_1st_id = list(p_info[p_info['first'] == 'lab']['p_id'])
lab_1st_data = pd.concat([yy_data[yy_data[p] == i.lower()] for i in lab_1st_id])
lab_1st_data = change_dframe_colnames(lab_1st_data, ['web', 'lab'], ['2nd', '1st'])

learning_data = pd.concat([web_1st_data, lab_1st_data])
learning_data = learning_data[[p, exp, ori, '1st', '2nd']]  # trim down
# add information for Bland-Altman plots
# add information for Bland-Altman plots
diff_label = '1st - 2nd'
mean_label = '(1st + 2nd) / 2'
learning_data[diff_label] = learning_data['1st'] - learning_data['2nd']
learning_data[mean_label] = np.mean([learning_data['1st'], learning_data['2nd']], 0)
# log transformed data to deal with with increased variance with increased magnitude
logdiff_label = 'log(1st) - log(2nd)'
logmean_label = '(log[1st] + log[2nd]) / 2 '
learning_data[logdiff_label] = np.log(learning_data['1st']) - np.log(learning_data['2nd'])
learning_data[logmean_label] = np.mean([np.log(learning_data['1st']), np.log(learning_data['2nd'])], 0)
ratio_label = 'threshold ratio (1st / 2nd)'
learning_data[ratio_label] = np.asarray(learning_data['1st']) / np.asarray(learning_data['2nd'])
learning = bland_altman(learning_data, '1st vs 2nd', hue=ori, labels=[diff_label, mean_label],
                        loglabels=[logdiff_label, logmean_label], ratiolabel=ratio_label,
                        save=True, show_ratio=True, show_log=True, show_linear=False, markerlist=None,
                        bias_adjust=False, checkifnormal=True, save_dir=gph_dir)
print(learning)
