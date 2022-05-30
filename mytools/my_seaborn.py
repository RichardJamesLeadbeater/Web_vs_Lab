import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def barplot_err(data, x, y, xerr=None, yerr=None, **kwargs):
    # barplot with custom SE (produced by +- the t_sem from single value)
    _data = []
    for _i in data.index:
        _data_i = pd.concat([data.loc[_i:_i]] * 3, ignore_index=True, sort=False)
        _row = data.loc[_i]
        if xerr is not None:
            _data_i[x] = [_row[x] - _row[xerr], _row[x], _row[x] + _row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    # _ax = sns.catplot(x=x, y=y, data=_data, ci='sd', kind='bar', **kwargs)
    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', **kwargs)
    return _ax


def load_seaborn_prefs(style="ticks", context="talk", custom_rc=None):
    if custom_rc is None:
        custom_rc = {}
    axes_color = [0.2, 0.2, 0.2, 0.95]
    rc_default = {'axes.edgecolor': axes_color, 'xtick.color': axes_color, 'ytick.color': axes_color,
                  'axes.linewidth': 1, 'legend.title_fontsize': 0, 'legend.fontsize': 13, 'patch.linewidth': 1.2,
                  'xtick.major.width': 1, 'xtick.minor.width': 1, 'ytick.major.width': 1, 'ytick.minor.width': 1,
                  'errorbar.capsize': 0.04, 'hatch.color': 'k', 'lines.linewidth': 2,
                  'axes.titlesize': 25, 'xtick.labelsize': 15, 'ytick.labelsize': 17,
                  'axes.labelsize': 20, 'font.family': 'sans-serif'}
    # update dict from kwargs entries
    for k in custom_rc:
        if rc_default.__contains__(k):
            rc_default[k] = custom_rc[k]
        else:
            print(f"KeyError: {k} is not a key in rc_default")
    # set rc_default params through seaborn
    sns.set_theme(style=style, context=context,
                  rc=rc_default)
    # matplotlib params set separately from seaborn
    defaults = {'figure.figsize': (6, 4), 'hatch.linewidth': 3.4}
    for i_param in ['figure.figsize', 'hatch.linewidth']:
        if custom_rc.__contains__(i_param):
            mpl.rcParams[i_param] = custom_rc[i_param]
        else:
            mpl.rcParams[i_param] = defaults[i_param]


def customise_barplot(barplot, title=None, ylabel=None, xlabel=None, y_major_ticks=None, y_minor_ticks=None, ylim=None,
                      hatch=True, len_x=1, legend=False, despine=True, tight=True, dpi=None):
    if title is not None:
        barplot.set_title(title)
    if ylabel is not None:
        barplot.set_ylabel(ylabel, fontweight='bold')
    if xlabel is not None:
        barplot.set_xlabel(xlabel, fontweight='bold')
    if y_major_ticks is not None:
        barplot.set_yticks(y_major_ticks)
        barplot.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        barplot.set_yticks(y_minor_ticks, minor=True)
    if ylim is not None:
        barplot.set_ylim(ylim)
    if hatch is True:
        hatches = [""] * len_x + [""] * len_x + ["\\"] * len_x + ["/"] * len_x
        for i, i_bar in enumerate(barplot.patches):
            i_bar.set_hatch(hatches[i])
            i_bar.set_edgecolor('w')
    if legend is False:
        barplot.get_legend().remove()
    if despine is True:
        sns.despine(top=True, right=True)
    if dpi is not None:
        barplot.figure.set_dpi(dpi)
    if tight is True:
        barplot.figure.tight_layout()


def plot_bar(dframe, x, y, hue, hue_order, palette, title='', y_label='', x_label='', legend=True,
             y_lim=None, y_major_ticks=None, y_minor_ticks=None,
             custom_err=False, custom_err_label='t_sem', save_name='', save_dir=None, fig_dims=(7, 4), my_dpi=100,
             close_plot=False):
    _fig, _ax = plt.subplots(figsize=fig_dims)
    if custom_err is False:
        bar_plot = sns.barplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                               errwidth=2, capsize=0.05, errcolor=[.1, .1, .1, 0.8],
                               ci=68, n_boot=2000, ax=_ax)
    else:
        bar_plot = barplot_err(data=dframe, x=x, y=y, yerr=custom_err_label, hue=hue, hue_order=hue_order, palette=palette,
                               errwidth=2, capsize=.05, errcolor=[.1, .1, .1, 0.8], ax=_ax)
    if legend is False:
        bar_plot.get_legend().remove()
    # PLOT LABELS #
    bar_plot.set_title(title, y=1.13, fontweight='bold')
    bar_plot.set_ylabel(y_label, size=20, fontweight='bold')
    bar_plot.set_xlabel(x_label, size=24, labelpad=9)
    # TICK LABELS #
    bar_plot.set_ylim(y_lim)
    if y_major_ticks is not None:
        bar_plot.set_yticks(y_major_ticks)
        bar_plot.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        bar_plot.set_yticks(y_minor_ticks, minor=True)
    # HATCHES #
    n_bars = len(dframe[x].unique())  # n bar clusters
    hatches = [""] * n_bars + [""] * n_bars + ["\\"] * n_bars + ["/"] * n_bars
    for idx, _bar in enumerate(bar_plot.patches):
        _bar.set_hatch(hatches[idx])
        _bar.set_edgecolor('w')
    # FIGURE LAYOUT #
    _fig.set_dpi(my_dpi)
    _fig.tight_layout()
    # SAVE FIGURE #
    if all(i for i in [save_dir, save_name]):
        bar_plot.figure.savefig(os.path.join(os.path.join(save_dir, save_name)))
    # close plot before continuing
    if close_plot is True:
        plt.close(_fig)
    return _fig, _ax


def plot_iqr(dframe, x, y, hue, hue_order=None, palette='colorblind', title='', x_label='', y_label='', legend=True,
             y_lim=None, y_major_ticks=None, y_minor_ticks=None,
             save_name='', save_dir=None, figure_dims=(7, 4), close_plot=True, my_dpi=100,
             include_strip=False):

    # interquartile range plots
    _fig, _ax = plt.subplots(figsize=figure_dims)
    # plot strip plot then overlay boxplot
    if include_strip is True:
        strip_plot = sns.stripplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order,
                                   palette=np.asarray(palette) / 2, linewidth=1.6,
                                   dodge=True, edgecolor=[1] * 3, alpha=0.5, size=6, jitter=.12,
                                   ax=_ax)  # alpha>0 to show
    # plot box plot with custom styles (median, boxprops)
    median_props = dict(linestyle='--', linewidth=2.2, color='k', alpha=1)
    mean_props = {"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black",
                  "markersize": 6}
    box_props = dict(linestyle='-', linewidth=1.4, edgecolor=[0.95] * 3, alpha=1)
    box_plot = sns.boxplot(data=dframe, x=x, y=y, hue=hue, hue_order=hue_order, palette=palette,
                           linewidth=0.8, showfliers=False, showmeans=True, meanprops=mean_props,
                           medianprops=median_props, boxprops=box_props, ax=_ax)
    if legend is False:
        _ax.get_legend().remove()  # remove legend (for reports screenshot)
    # PLOT LABELS #
    _ax.set_title(title, y=1.13, fontweight='bold')
    _ax.set_ylabel(y_label, size=20, fontweight='bold')
    _ax.set_xlabel(x_label, size=24, labelpad=9)
    # TICK LABELS #
    _ax.set_ylim(y_lim)
    if y_major_ticks is not None:
        _ax.set_yticks(y_major_ticks)
        _ax.set_yticklabels([f"{int(i)}" for i in y_major_ticks])
    if y_minor_ticks is not None:
        _ax.set_yticks(y_minor_ticks, minor=True)
    # HATCHES #
    n_bars = len(dframe[x].unique())  # n bar clusters
    hatches = [""] * n_bars + [""] * n_bars + ["\\"] * n_bars + ["/"] * n_bars
    for hatch, patch in zip(hatches, _ax.artists):
        patch.set_hatch(hatch)
    # FIGURE LAYOUT #
    sns.despine(_fig, _ax, top=True, right=True, offset=5)
    _fig.set_dpi(my_dpi)
    _fig.tight_layout()
    # SAVE FIGURE #
    if all(i for i in [save_dir, save_name]):
        _fig.savefig(os.path.join(os.path.join(save_dir, save_name)))
    # close plot before continuing
    if close_plot is True:
        plt.close(_fig)
    return _fig, _ax
