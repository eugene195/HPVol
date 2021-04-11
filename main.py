# Dear Jupyter User,
# Don't forget these two:
# !pip install tick
# !pip install py_vollib_vectorized

import pandas as pd
import numpy as np
from lib import raw_ds_columns, smoothen_ts_jumps, calc_ts_diff
from visualization import display_point_process_events, plot_ts

from tick.hawkes import HawkesSumExpKern

ticks_atm_log_mon = pd.read_csv("data/current.csv", header=None, names=raw_ds_columns())
ticks_atm_log_mon = ticks_atm_log_mon.iloc[1:]

ticks_atm_log_mon["Bid_time"] = ticks_atm_log_mon["Bid_time"].astype(int)
ticks_atm_log_mon["Mid_IV"] = ticks_atm_log_mon["Mid_IV"].astype(float)

time_start = min(ticks_atm_log_mon["Bid_time"])
ticks_atm_log_mon["Bid_time"] = ticks_atm_log_mon["Bid_time"] - time_start
ticks_atm_log_mon = ticks_atm_log_mon.sort_values("Bid_time")

# 479 is prime (I chop the tail off because 500 events is enough for testing purposes)
ticks_atm_log_mon = ticks_atm_log_mon.iloc[:479]
selected_strikes = sorted(list(set(ticks_atm_log_mon['Price_strike'])))

smoothen_keys = ["Mid_IV"]

smoothened_slices = pd.concat([
    smoothen_ts_jumps(
        ticks_atm_log_mon.loc[ticks_atm_log_mon['Price_strike'] == strike], key=smoothen_keys[0]
    )
    for strike in selected_strikes
]).sort_values("Bid_time")

smoothened_ts = calc_ts_diff(
    smoothen_ts_jumps(smoothened_slices, key=smoothen_keys[0]).sort_values("Bid_time"),
    keys=smoothen_keys + ["Bid_time"]
)
smoothened_ts["Sign"] = np.sign(smoothened_ts["{}_diff".format(smoothen_keys[0])])
n_plus_ts, n_minus_ts = smoothened_ts.loc[smoothened_ts["Sign"] > 0], \
                        smoothened_ts.loc[smoothened_ts["Sign"] < 0]

display_point_process_events(n_plus_ts)
display_point_process_events(n_minus_ts)
plot_ts(smoothened_ts, smoothen_keys[0])


def get_learner(decays, verbose=False):
    return HawkesSumExpKern(decays, verbose=verbose, tol=1e-10, max_iter=1000)


def get_HP_exp_learner(decays, events, verbose=False):
    learner = get_learner(decays, verbose)
    learner.fit(events)
    if verbose:
        print("Adj: {}, decays: {}, baselines: {}".format(learner.adjacency, decays, learner.baseline))
    return learner


def fun(decays, events):
    return - HawkesSumExpKern(decays=decays, penalty="elasticnet", elastic_net_ratio=0.9, solver="agd",
                              max_iter=1000).fit(events).score()

# fixme: this file is not used
