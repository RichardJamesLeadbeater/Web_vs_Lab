import numpy as np
from scipy.optimize import curve_fit
import pylab
import os


def logistic_func(x, a, b, intercept=0.5):
    # x=xval    a=midpoint    b=slope
    # +0.5 to intercept since it spans 0.5 to 1
    maxval = 1 - intercept
    y = intercept + maxval / (1 + np.exp(-(x - a) / b))
    return y


def fit_logistic(x, y, maxfev=4000, p0=None, bounds=(-np.inf, np.inf)):
    # set default
    if p0 is None:
        p0 = [np.mean(x), 3]  # starting parameters for curve fit
    if bounds == (-np.inf, np.inf):
        bounds = ([0, 0], [x.max(), 10])  # bounds for threshold and slope (higher vals = more gradual slope)

    x = np.asarray(x)
    y = np.asarray(y)
    # determine params of logistic curve using least-squares method
    popt, pcov = curve_fit(logistic_func, x, y,
                           method='trf',
                           maxfev=maxfev, p0=p0, bounds=bounds
                           )
    # popt, pcov = curve_fit(logistic_function, x_vals, y_vals, method='lm', maxfev=4000)

    thresh_opt = popt[0]
    thresh_std = np.sqrt(np.diag(pcov))[0]
    slope_opt = popt[1]
    slope_std = np.sqrt(np.diag(pcov))[1]

    # output of logistic function given x-vals and optimised params
    x = np.linspace(x.min(), x.max(), len(x) * 1000)
    y = logistic_func(x, *popt)

    # namedtuple does not work with multiprocessing
    return {'x': x, 'y': y, 'threshold': thresh_opt, 'slope': slope_opt,
            'threshold_std': thresh_std, 'slope_std': slope_std}


def plot_psychometric(x_axis, y_axis, x_fit_axis, y_fit_axis, threshold_value, title='', close_all=True,
                      to_save=False, save_dir=None, save_name='', percent=False, legend=True):
    threshold_idx = np.argmin(abs(x_fit_axis - threshold_value))
    if close_all:
        pylab.close('all')
    else:
        pass
    if percent is True:  # convert to percentage units
        y_axis *= 100
        y_fit_axis *= 100
        y_label = 'Percent Correct (%)'
        y_lim = (0, 102)
    else:
        y_label = 'Proportion Correct'
        y_lim = (0, 1.02)
    fig = pylab.figure(figsize=(4, 3))
    pylab.plot(x_axis, y_axis, 'o', label='data')
    pylab.plot(x_fit_axis, y_fit_axis, label='fit')
    # pylab.ylim(0 - .01, 1.01)
    # pylab.xlim(x_axis.min() - 0.01, x_axis.max() + 0.01)
    pylab.plot([threshold_value, threshold_value], [0, y_fit_axis[threshold_idx]], 'k--', alpha=0.3)
    pylab.plot([0, threshold_value], [y_fit_axis[threshold_idx], y_fit_axis[threshold_idx]], 'k--', alpha=0.3)
    # adjust plot
    ax = fig.axes[0]
    ax.set(ylim=y_lim, xlim=(x_axis.min(), x_axis.max() + 0.02))
    ax.set_xlabel("Orientation Difference ($^\circ$)", fontweight='bold', fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel(y_label, fontweight='bold', fontsize=12, fontfamily='sans-serif')
    # ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=10)
    # ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=10)
    pylab.title(title, pad=20, fontsize=13)
    if legend is True:
        pylab.legend(loc='best')
    fig.set_dpi(150)
    fig.tight_layout()
    if to_save is True or save_dir is not None:
        pylab.savefig(os.path.join(save_dir, f"{save_name}.png"))
