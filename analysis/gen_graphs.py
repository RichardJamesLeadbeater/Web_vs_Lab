import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from Code.mytools.filenav import browser
from Code.mytools.dframe_tools import change_dframe_labels
from Code.mytools.my_seaborn import barplot_err, load_seaborn_prefs, customise_barplot


def plot_indiv_bar(dframe, measure, title_cond, iv1, iv2=None, title=None, savepath=None, forprinting=False,
                   ylims=None, ylim_log=0.01, col_wrap=3, uselog=False, close_plot=True):
    data_plot = dframe.copy()  # preserve original

    vals = [title_cond[0], iv1[0]]
    slicer = [title_cond[1], iv1[1]]  # which conds to plot
    label = [title_cond[2], iv1[2]]

    # create constant indices for consistent use in function
    tit_idx = 0
    iv1_idx = 1

    # index vals to be plotted... but still retain all vals of iv as it determines color wheel
    plot_vals = [title_cond[tit_idx], vals[iv1_idx][slicer[iv1_idx]]]

    if iv2:
        vals.append(iv2[0])
        slicer.append(iv2[1])
        label.append(iv2[2])
        iv2_idx = 2
        plot_vals.append(vals[iv2_idx][slicer[iv2_idx]])

    colour_palette = manual_color
    colour_order = vals[iv1_idx]
    colour_order = colour_order[slicer[iv1_idx]]

    # creates dataframe made up of only values for plotting
    data_plot = pd.concat(data_plot[(data_plot[label[tit_idx]] == x)] for x in plot_vals[tit_idx])
    data_plot = pd.concat(data_plot[(data_plot[label[iv1_idx]] == x)] for x in plot_vals[iv1_idx])
    if iv2:
        data_plot = pd.concat(data_plot[(data_plot[label[iv2_idx]] == x)] for x in plot_vals[iv2_idx])

    load_seaborn_prefs()
    # plots multiple plots on facegrid for each col (title_cond)
    if iv2:  # RAW DATA PLOTS
        # i_bar = sns.catplot(x=label[iv2_idx], y=measure,
        #                 hue=label[iv1_idx], hue_order=colour_order, kind='bar', legend=False,
        #                 errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8],
        #                 # sharey=True, sharex=True,
        #                 data=data_plot, ci=68, palette=colour_palette,
        #                 col=label[tit_idx], col_wrap=3)

        # force custom error bars onto sns.barplot
        fig_dims = (4.5, 3)
        fig, ax = plt.subplots(figsize=fig_dims)
        i_bar = barplot_err(x=label[iv2_idx], y=measure, xerr=None, yerr='t_sem',
                            hue=label[iv1_idx], data=data_plot, hue_order=colour_order,
                            errwidth=1.2, capsize=.04, errcolor=[.2, .2, .2, 0.8],
                            # sharey=True, sharex=True,
                            palette=colour_palette, ax=ax
                            )
        i_bar.get_legend().remove()
        i_bar.set(ylim=ylims)
        # very hacky but works
        n_bar = len(data_plot[label[iv2_idx]].unique())  # count number of bar groups
        hatches = [""] * n_bar + [""] * n_bar + ["\\"] * n_bar + ["/"] * n_bar
        # hatches = ["", "", "", "", "\\", "\\", "/", "/"]
        for idx, thisbar in enumerate(i_bar.patches):
            thisbar.set_hatch(hatches[idx])
            thisbar.set_edgecolor('w')
        # i_bar.set(xlabel=label[iv2_idx], ylim=(0, 18))
        i_bar.set_title(title.split('_')[0], y=0.8, size=label_size * 1.15, fontweight='bold')
        i_bar.set_ylabel('threshold ($^\circ$)', size=20, fontweight='bold')
        i_bar.set_xlabel('task', size=24, labelpad=9)
        i_bar.set_xticklabels(i_bar.get_xticklabels(), size=20, fontweight='bold', y=.03)
        i_bar.set_yticks([0, 10, 20])
        i_bar.set_yticklabels([0, 10, 20], size=label_size, fontweight='bold')
        y_majorticks = np.linspace(0, 20, 3)
        ax.set_yticks(y_majorticks, minor=False)
        ax.set_yticklabels([f"{int(i)}" for i in y_majorticks])
        y_minorticks = np.linspace(5, 25, 3)
        ax.set_yticks(y_minorticks, minor=True)
        i_bar.spines['top'].set_visible(False)
        i_bar.spines['right'].set_visible(False)
        i_bar.figure.tight_layout()
        # i_bar.figure.set(dpi=my_dpi)
        i_bar.figure.savefig(os.path.join(savepath, f"{title}.png"))
        if close_plot is True:
            plt.close()


def get_ylim_logscale(dframe, dframe_dv='mean'):
    # scales differently if no heavy decimals
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


def plot_mocs_data(data, x, y, hue, hue_order, palette, title='', y_label='', x_label='',
                   y_lim=None, figure_dims=(7, 4), y_major_ticks=None, y_minor_ticks=None,
                   custom_err=None, save_name='', save_dir=None, close_plot=True):
    # barplot
    fig, ax = plt.subplots(figsize=figure_dims)  # i
    # plot
    if custom_err:
        # plots custom error bars
        barplot = barplot_err(data=data, x=x, y=y, yerr='t_sem', hue=hue, hue_order=hue_order, palette=palette,
                              errwidth=2, capsize=.05, errcolor=[.1, .1, .1, 0.8], ax=ax)
    else:
        barplot = sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                              errwidth=2, capsize=.05, errcolor=[.1, .1, .1, 0.8],
                              ci=68, n_boot=2000, ax=ax)
    # perform like this to prevent resetting ticklabels
    barplot.set_title(title, fontweight='bold')
    barplot.set_ylabel(y_label, fontweight='bold')
    barplot.set_xlabel(x_label, fontweight='bold')
    barplot.set_ylim(y_lim)
    # change visible tick labels (this way prevents rescaling of axes)
    if y_major_ticks is not None:
        barplot.set_yticks(y_major_ticks)
        barplot.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        barplot.set_yticks(y_minor_ticks, minor=True)

    # set hatches to oblique bars (cardinals defined by colour)
    hatches = ["", "", "", "", "\\", "\\", "/", "/"]
    for i, i_bar in enumerate(barplot.patches):
        i_bar.set_hatch(hatches[i])
        i_bar.set_edgecolor('w')
    # remove legend (screenshot to set custom position of legend in reports)
    barplot.get_legend().remove()
    # formatting
    sns.despine(top=True, right=True)  # remove right and top border
    barplot.figure.tight_layout()
    barplot.figure.set_dpi(my_dpi)
    # save figure
    if not save_dir:
        save_dir = os.getcwd()
    barplot.figure.savefig(os.path.join(os.path.join(save_dir, save_name)))
    # close plot before continuing
    if close_plot:
        plt.close()


def plot_iqr(data, x, y, hue, palette, x_label='', y_label='',
             save_name='', save_dir=None, figure_dims=(7, 4), y_lim=None, close_plot=True):


    # interquartile range plots
    fig, ax = plt.subplots(figsize=figure_dims)
    # plot strip plot then overlay boxplot
    strip_plot = sns.stripplot(data=data, x=x, y=y, hue=hue,
                               palette=np.asarray(manual_color) / 2, linewidth=1.6,
                               dodge=True, edgecolor=[1] * 3, alpha=0.0, size=6, jitter=.12)  # alpha>0 to show
    # plot box plot with custom styles (median, boxprops)
    median_props = dict(linestyle='--', linewidth=2.2, color='k', alpha=1)
    mean_props = {"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black",
                  "markersize": 6}
    box_props = dict(linestyle='-', linewidth=1.4, edgecolor=[0.95] * 3, alpha=1)
    box_plot = sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette,
                           linewidth=0.8, showfliers=False, showmeans=True, meanprops=mean_props,
                           medianprops=median_props, boxprops=box_props, ax=ax)

    ax.get_legend().remove()  # remove legend (for reports screenshot)
    ax.set_ylim(y_lim)
    y_majorticks = np.linspace(0, 25, 6)
    ax.set_yticks(y_majorticks, minor=False)
    ax.set_yticklabels([f"{int(i)}" for i in y_majorticks])
    y_minorticks = np.linspace(2.5, 27.5, 6)
    ax.set_yticks(y_minorticks, minor=True)
    ax.set_xlabel(x_label, fontweight='bold', size=22)
    ax.set_ylabel(y_label, fontweight='bold')
    # set hatches in each box
    hatches = ["", "", "\\", "/", "", "", "\\", "/"]
    for hatch, patch in zip(hatches, ax.artists):
        patch.set_hatch(hatch)
    sns.despine(fig, ax, top=True, right=True, offset=5)
    ax.figure.tight_layout()
    ax.figure.set_dpi(my_dpi)
    if not save_dir:
        save_dir = os.getcwd()
    ax.figure.savefig(os.path.join(os.path.join(save_dir, save_name)))
    # close plot before continuing
    if close_plot:
        plt.close()


def plot_violin(data, x, y, hue, hue_order, palette, title='', y_label='', x_label='',
                y_lim=None, figure_dims=(7, 4), y_major_ticks=None, y_minor_ticks=None,
                save_name='', save_dir=None, close_plot=True):
    # barplot
    fig, ax = plt.subplots(figsize=figure_dims)  # i
    # plot
    vioplot = sns.violinplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                             ax=ax, inner=None, scale='area')
    # perform like this to prevent resetting ticklabels
    vioplot.set_title(title, fontweight='bold')
    vioplot.set_ylabel(y_label, fontweight='bold')
    vioplot.set_xlabel(x_label, fontweight='bold')
    vioplot.set_ylim(y_lim)
    # change visible tick labels (this way prevents rescaling of axes)
    y_majorticks = np.linspace(0, 25, 6)
    ax.set_yticks(y_majorticks, minor=False)
    ax.set_yticklabels([f"{int(i)}" for i in y_majorticks])
    y_minorticks = np.linspace(2.5, 27.5, 6)
    ax.set_yticks(y_minorticks, minor=True)

    # set hatches to oblique bars (cardinals defined by colour)
    hatches = ["", "", "", "", "\\", "\\", "/", "/"]
    for i, i_bar in enumerate(vioplot.patches):
        i_bar.set_hatch(hatches[i])
        i_bar.set_edgecolor('w')
    # remove legend (screenshot to set custom position of legend in reports)
    vioplot.get_legend().remove()
    # formatting
    sns.despine(fig, ax, top=True, right=True, offset=15)  # remove right and top border
    vioplot.figure.tight_layout()
    vioplot.figure.set_dpi(my_dpi)
    # save figure
    if not save_dir:
        save_dir = os.getcwd()
    vioplot.figure.savefig(os.path.join(os.path.join(save_dir, save_name)))
    # close plot before continuing
    if close_plot:
        plt.close()
    # interquartile range plots
    fig, ax = plt.subplots(figsize=figure_dims)
    # # plot vioplot
    # vio_plot = sns.violinplot(data=data, x=x, y=y, hue=hue, palette=palette,
    #                           )
    # ax.get_legend().remove()  # remove legend (for reports screenshot)
    # ax.set_ylim(y_lim)
    # y_majorticks = np.linspace(0, 25, 6)
    # ax.set_yticks(y_majorticks, minor=False)
    # ax.set_yticklabels([f"{int(i)}" for i in y_majorticks])
    # y_minorticks = np.linspace(2.5, 27.5, 6)
    # ax.set_yticks(y_minorticks, minor=True)
    # ax.set_xlabel(x_label, fontweight='bold', size=22)
    # ax.set_ylabel(y_label, fontweight='bold')
    # # set hatches in each box
    # hatches = ["", "", "\\", "/", "", "", "\\", "/"]
    # for hatch, patch in zip(hatches, ax.artists):
    #     patch.set_hatch(hatch)
    # ax.figure.set_dpi(my_dpi)
    # sns.despine(top=True, right=True)
    # ax.figure.tight_layout()
    # if not save_dir:
    #     save_dir = os.getcwd()
    # ax.figure.savefig(os.path.join(os.path.join(save_dir, save_name)))


# init paths
# og_path = browser(False, True, chdir=True)
og_path = os.getcwd()
sum_path = os.path.join(og_path, 'summary')
gph_path = os.path.join(og_path, 'graphs')
indiv_gph_path = os.path.join(gph_path, 'indiv')
means_gph_path = os.path.join(gph_path, 'means')
iqr_gph_path = os.path.join(gph_path, 'iqr')
vio_gph_path = os.path.join(gph_path, 'violin')
for ipath in [gph_path, indiv_gph_path, means_gph_path, iqr_gph_path, vio_gph_path]:
    if not os.path.exists(ipath):
        os.makedirs(ipath)

# set information about the experiment
exp_info = pd.read_csv('expinfo.csv')
exp_name = exp_info['exp_name'].tolist()[0]
iv_name = exp_info['iv'].to_list()[0]

summary_data = pd.read_csv(os.path.join(sum_path, f"{exp_name}_bootstrap_allsummary.csv"))
# summary_data = pd.read_csv(os.path.join(sum_path, f"{exp_name}_allsummary.csv"))

print(f"Plotting data from: {exp_name}")
# general plot params
my_dpi = 125

# labels consistent with WvL data
p = 'observer'
ori = 'ori'
exp = 'task'
measure = 'threshold'
env = 'enviro'

# initialise plot data as copy of summary_data
plot_data = summary_data.copy()

# capitalise initials for graphs
plot_data = change_dframe_labels(plot_data, p, plot_data[p].unique().tolist(),
                                 [i.upper() for i in plot_data[p].unique().tolist()]
                                 )
# ordered list of each ori label and rename for deg sign
eachori_lbl = ['0$^\circ$ (H)', '90$^\circ$ (V)', '- 45$^\circ$', '+ 45$^\circ$', 'Cardinal', 'Oblique']
plot_data = change_dframe_labels(plot_data, ori,
                                 plot_data[ori].unique().tolist(), eachori_lbl)  # change labels for graphs
# set colors assoc'd w each ori
color_order = eachori_lbl + ['Cardinal', 'Oblique']
cblind = sns.color_palette('colorblind')
manual_color = [[0.2] * 3, [0.8] * 3, cblind[2], cblind[3]]  # set H and V to B&W (hashes look unclear)
# create params for plot (data, name, color_order, palette, slice)
plot_data = plot_data[(plot_data[ori] != 'Cardinal') & (plot_data[ori] != 'Oblique')]
plot_name = 'std_oris'
color_order = color_order[0:4]
palette = manual_color

# load info about WvL subsample
p_info = pd.read_csv('/Users/lpxrl4/Documents/Code Repositories/Psychophysics/Code/Data Analysis/datapull/Home_vs_Lab/WvL/subsample_ids.csv')

# draw graphs from web and lab data separately
# loops through three ways of splitting up data then runs code on each
for weborlab in ['allweb', 'web', 'lab']:
    i_data = summary_data.copy()
    iexpname = f"{exp_name}_{weborlab}"
    if weborlab == 'web':
        # names of custom p1 p2 etc. which are only used in 'web' subsample
        p_list = list(p_info['p_id'])
        subsampledata = pd.concat([i_data[(i_data[p] == i)] for i in p_list])
        i_data = subsampledata[subsampledata[env] == 'web']
        figdims = (7, 4)
    elif weborlab == 'allweb':
        i_data = i_data[(i_data[env] == 'web')]
        figdims = (8, 4)
    elif weborlab == 'lab':
        i_data = i_data[(i_data[env] == 'lab')]
        figdims = (7, 4)

    # convert observer names to anonymised form
    if any(weborlab == i for i in ['web', 'lab']):
        i_data = change_dframe_labels(i_data, col=p, old_labels=sorted(i_data[p].unique().tolist()),
                                      new_labels=[f"p{i + 1}" for i in range(len(observer_vals))])
    observer_vals = sorted(i_data[p].unique().tolist())
    iv_vals = i_data[iv_name].unique().tolist()
    # capitalise initials for graphs
    i_data = change_dframe_labels(i_data, p, observer_vals, [i.upper() for i in observer_vals])
    # ordered list of each ori label and rename for deg sign
    eachori_lbl = ['0$^\circ$ (H)', '90$^\circ$ (V)', '- 45$^\circ$', '+ 45$^\circ$', 'Cardinal', 'Oblique']
    i_data = change_dframe_labels(i_data, ori, i_data[ori].unique().tolist(), eachori_lbl)  # change labels for graphs
    # set colors assoc'd w each ori
    color_order = eachori_lbl + ['Cardinal', 'Oblique']
    cblind = sns.color_palette('colorblind')
    manual_color = [[0.2] * 3, [0.8] * 3, cblind[2], cblind[3]]  # set H and V to B&W (hashes look unclear)
    # create params for plot (data, name, color_order, palette, slice)
    i_data = i_data[(i_data[ori] != 'Cardinal') & (i_data[ori] != 'Oblique')]
    plot_name = 'std_oris'
    color_order = color_order[0:4]
    palette = manual_color
    plot_slice = slice(0, 4)
    # init seaborn settings
    load_seaborn_prefs(context='paper')
    legend_fontsize = 18
    label_size = 20
    tick_size = 15
    title_size = 19
    width_space = .4
    height_space = .6
    my_dpi = 125
    # plot either pooled data (summary_data) or data for each participant (raw)
    print('# # #\n[which_graph, i_plot_data, i_plot_name, i_slice] have not been set\n# # #')
    which_graph = ''
    i_plot_name = ''
    i_slice = None
    i_plot_data = None

    # plot graphs from summary_data data
    y_limit = (0, 27.5)
    y_label = 'mean threshold ($^\circ$)'
    x_label = 'task'
    # IQR of summary_data data
    plot_iqr(i_data, x=iv_name, y=measure, hue=ori, palette=palette,
             x_label=x_label, y_label=y_label, save_name=f"{iexpname}_iqr",
             save_dir=iqr_gph_path, y_lim=y_limit, figure_dims=(7, 4))
    # violin of summary data
    plot_violin(i_data, x=iv_name, y=measure, hue=ori, hue_order=color_order, palette=palette,
                y_label=y_label, x_label=x_label, y_lim=y_limit, save_name=f"{iexpname}_violin",
                save_dir=vio_gph_path, close_plot=True)

    plot_mocs_data(i_data, x=iv_name, y=measure, hue=ori, hue_order=color_order, palette=palette,
                   y_label=y_label, x_label=x_label, y_lim=y_limit, save_name=f"{iexpname}_bar",
                   save_dir=means_gph_path, close_plot=False)



    # plot graphs from raw data (e.g. individual participants)
    if weborlab != 'allweb':
        # perform separate plots for each participant (easier use for write-up)
        for i_p in i_data[p].unique():
            p_plot_data = i_data[(i_data[p] == i_p)]
            if len(p_plot_data[exp].unique()) == 1:
                blank = p_plot_data.copy()
                blank['task'] = 'spatial'
                blank['threshold'] = np.nan
                blank['t_sem'] = np.nan
                p_plot_data = pd.concat([p_plot_data, blank]).reset_index(drop=True)
            # below code is old and disgusting - never reuse
            plot_indiv_bar(p_plot_data, measure=measure,
                           title_cond=[[i.upper() for i in observer_vals], slice(0, len(observer_vals)), p],
                           iv1=[eachori_lbl, slice(0, 4), ori], iv2=[iv_vals, slice(0, len(iv_vals)), iv_name],
                           title=f"{i_p}_{i_plot_name}_{exp_name}_{weborlab}", savepath=indiv_gph_path,
                           forprinting=False, ylims=[0, 25], ylim_log=None, col_wrap=1, uselog=False)
    # plt.close()

print('debug')
