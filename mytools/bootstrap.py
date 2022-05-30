import itertools as itl
from itertools import repeat
import multiprocessing as mlp
from collections import namedtuple
import numpy as np
import random
from .multiproc import run_function_and_unpack_args_kwargs


def resample(dataset, n_samples=1000):
    dataset = np.asarray(dataset)
    # pull out resampled data for n_samples
    resampled = [None] * n_samples
    for i in range(n_samples):
        resampled[i] = random.choices(dataset, k=len(dataset))
    return np.asarray(resampled)


def bs_stats(resampled_means, name='output'):
    # input resampled means
    Output = namedtuple(name, ['stdev', 'sem', 'ci95', 'mean'],
                        defaults=[np.var(resampled_means, ddof=1), np.std(resampled_means, ddof=1),
                                  np.percentile(resampled_means, [2.5, 97.5]), np.mean(resampled_means)])
    return Output()


def bootstrap_logistic_fit(fun, x_data, y_data, use_mp=False):
    # args e.g. (x_vals, y_vals) input as list to be used in next function
    if use_mp:
        x_data = [x_data] * len(y_data)  # make same length for iteration in multiprocessing
        # use multiprocessing with fit function
        with mlp.Pool() as pool:
            # resampling with og data for n_samples (zip keeps items from each list in pairs)
            resamp_data = pool.starmap(fun, zip(x_data, y_data))
            pool.close()
            pool.join()
    else:
        resamp_data = list(itl.starmap(fun, zip(repeat(x_data), y_data)))

    # calc sem from all thresholds / slopes from each resampled dataset
    resamp_thresh = bs_stats([i['threshold'] for i in resamp_data], name='threshold')
    resamp_slope = bs_stats([i['slope'] for i in resamp_data], name='slope')

    return resamp_thresh, resamp_slope
