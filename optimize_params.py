from functools import partial

from tick.hawkes import HawkesExpKern, HawkesSumExpKern
import numpy as np
from scipy.optimize import *

from lib import csv_reader
from multiprocessing_sim import split_simple_config, try_params, run_test, split_complex_config


def single_exp(decays, events):
    return - HawkesExpKern(decays=decays[0], penalty='elasticnet', tol=1e-8,
                           elastic_net_ratio=0.9, max_iter=1000).fit(events).score()


def sum_exp(decays, events):
    return - HawkesSumExpKern(decays=decays, penalty='elasticnet', tol=1e-8,
                              elastic_net_ratio=0.9, max_iter=1000).fit(events).score()


def sum_n_exp_minimiser(decays, timestamps):
    return minimize(sum_exp, x0=decays, args=(timestamps), method='Nelder-Mead', tol=1e-5)


def exp_minimiser(x, timestamps):
    return minimize(single_exp, x0=[x], args=(timestamps), method='Nelder-Mead', tol=1e-5)


def test_1(timestamps):
    start = 0.001
    stop = 5
    N = 1e6
    test_name = "1Expo"
    test1_params = np.linspace(start, stop, int(N))
    param_configs = [(c, "{}".format(c[0])) for c in split_simple_config(test1_params)]
    run_simulation_func = partial(try_params, timestamps, exp_minimiser, test_name)
    run_test(param_configs, run_simulation_func, test_name=test_name, parallel=True)


def test_2(timestamps):
    start = 0.001
    stop = 5
    N = 1e4
    test_name = "2Expo"
    test1_params = np.linspace(start, stop, int(N))
    param_configs = [(c, "{}".format(c[0])) for c in split_complex_config([test1_params] * 2)]
    run_simulation_func = partial(try_params, timestamps, sum_n_exp_minimiser, test_name)
    run_test(param_configs, run_simulation_func, test_name=test_name, parallel=True)


def get_timestamps():
    n_minus_ts_ticks_25dc = csv_reader('data/n_plus_ts_ticks_60dc_AAPL.csv')
    n_minus_ts_ticks_25dp = csv_reader('data/n_plus_ts_ticks_60dp_AAPL.csv')
    n_minus_ts_ticks_50dc = csv_reader('data/n_minus_ts_ticks_50dc_AAPL.csv')

    n_plus_ts_ticks_25dc = csv_reader('data/n_plus_ts_ticks_60dc_AAPL.csv')
    n_plus_ts_ticks_25dp = csv_reader('data/n_plus_ts_ticks_60dp_AAPL.csv')
    n_plus_ts_ticks_50dc = csv_reader('data/n_plus_ts_ticks_50dc_AAPL.csv')

    all_data = [
        n_plus_ts_ticks_25dc, n_minus_ts_ticks_25dc,
        n_plus_ts_ticks_50dc, n_minus_ts_ticks_50dc,
        n_plus_ts_ticks_25dp, n_minus_ts_ticks_25dp,
    ]

    all_timestamps = []
    for df in all_data:
        all_timestamps.append(np.array(list(df['Bid_time'])))

    max_min_time = max([min(ts_df) for ts_df in all_timestamps])
    min_max_time = min([max(ts_df) for ts_df in all_timestamps])

    trimmed_ts = [
        ts[(max_min_time < ts) & (ts < min_max_time)] - max_min_time
        for ts in all_timestamps
    ]

    for ts_ in trimmed_ts:
        print(ts_[0])
        print("Increasing: {}".format(np.all(np.diff(ts_) > 0)))

    return all_timestamps

test_1(
    get_timestamps()
)

test_2(
    get_timestamps()
)