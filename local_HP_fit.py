import pandas as pd
import numpy as np
from lib import raw_ds_columns, calc_ts_diff
from visualization import display_point_process_events
from scipy.optimize import minimize
from tick.hawkes import HawkesSumExpKern, HawkesBasisKernels, HawkesKernelTimeFunc, SimuHawkes
import statsmodels
from scipy import integrate
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import statsmodels.api as sm
import itertools

plt.style.use('ggplot')

import numpy as np
import pandas as pd
from tick.hawkes import *
from scipy.optimize import *
from tick.plot import plot_hawkes_kernels


def csv_reader(name):
    df = pd.read_csv("data/{}".format(name), header=None)

    df.set_index(0)

    df.drop(df.columns[0], axis=1, inplace=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    df["Bid_time"] = df["Bid_time"].astype(float)
    df["Mid_IV"] = df["Mid_IV"].astype(float)
    df["Mid_price"] = df["Mid_price"].astype(float)
    df = df.loc[df["Bid_time"] < 3e9]
    return df

n_minus_ts_ticks_25dc = csv_reader('n_minus_ts_ticks_25dc_sp100.csv')
n_minus_ts_ticks_25dp = csv_reader('n_minus_ts_ticks_25dp_sp100.csv')
n_minus_ts_ticks_50dc = csv_reader('n_minus_ts_ticks_50dc_sp100.csv')

n_plus_ts_ticks_25dc = csv_reader('n_plus_ts_ticks_25dc_sp100.csv')
n_plus_ts_ticks_25dp = csv_reader('n_plus_ts_ticks_25dp_sp100.csv')
n_plus_ts_ticks_50dc = csv_reader('n_plus_ts_ticks_50dc_sp100.csv')

all_data = [n_plus_ts_ticks_25dc, n_minus_ts_ticks_25dc,
            n_plus_ts_ticks_50dc, n_minus_ts_ticks_50dc,
            n_plus_ts_ticks_25dp, n_minus_ts_ticks_25dp, ]

all_timestamps = []
for df in all_data:
    all_timestamps.append(np.array(list(df['Bid_time'])))


def fun(decays, events=all_timestamps):
    return - HawkesExpKern(decays=decays[0], penalty='elasticnet', tol=1e-8,
                           elastic_net_ratio=0.9, max_iter=1000).fit(events).score()


best_beta = [0.18784562]
BEST_BETA = best_beta[0]

learner = HawkesExpKern(decays=BEST_BETA,tol=1e-10, penalty='elasticnet',
                          elastic_net_ratio=0.9, max_iter=2000)

learner.fit(all_timestamps)

alphas = learner.adjacency
betas = learner.decays
baseline = learner.baseline
loglik = learner.score()

print("Alphas: ")
print(np.round(alphas,4))

print("\n Baseline: ")
print(baseline)

print("\n Endogeneity:")
print(max(np.linalg.eigvals(alphas)))

print("\n Likelihood: ")
print(loglik)


def resid(x, intensities, timestamps, dim, method):
    print(dim)
    arrivals = timestamps[dim]
    thetas = np.zeros(len(arrivals) - 1)
    ints = intensities[dim]
    for i in range(1, len(arrivals)):
        mask = (x <= arrivals[i]) & (x >= arrivals[i - 1])
        xs = x[mask]
        ys = ints[mask]
        try:
            thetas[i - 1] = method(ys, xs)
        except:
            thetas[i - 1] = np.nan

    return thetas

def goodness_of_fit_par(learner, arrivals, step, method):
    dimension = learner.n_nodes
    intensities = learner.estimated_intensity(arrivals, step)[0]
    x = learner.estimated_intensity(arrivals, step)[1]
    residuals = [resid(x, intensities, arrivals, dim, method) for dim in range(dimension)]
    return residuals

def ks_test(resid):
    for res in resid:
        print(stats.kstest(res[np.logical_not(np.isnan(res))], 'expon'))

def plot_resid(resid, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Goodness-of-fit for nonparametric HP')

    for ax, res in zip(axes, resid):
        k = stats.probplot(res, dist=stats.expon, fit=True, plot=ax, rvalue=False)
        ax.plot(k[0][0], k[0][0], 'k--')

def ks_test(resid):
    return [
        stats.kstest(res[np.logical_not(np.isnan(res))], 'expon')
        for res in resid
    ]

def lb_test(resid):
    return [
        sm.stats.acorr_ljungbox(res[np.logical_not(np.isnan(res))], lags=[10], return_df=True)
        for res in resid
    ]

def ed_test(resid):
    results = []
    for res in resid:
        results.append(
            np.sqrt(len(res)) * (np.var(res, ddof=1) - 1) / np.sqrt(8)
        )
    return results

residuals = goodness_of_fit_par(learner, all_timestamps, 150, integrate.simps)
print(residuals)

