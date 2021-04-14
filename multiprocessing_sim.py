import glob
import itertools
import multiprocessing
import os
import traceback
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd


def _clean_directory(test_name):
    files = glob.glob("data/{}_Params_*".format(test_name))
    for f in files:
        os.remove(f)


def write_results_(best_initial_value, best_betas, best_result, file_name):
    if (best_initial_value == -np.inf) or (best_betas == -np.inf) or (best_result == -np.inf):
        results = {"init": np.nan, "betas": np.nan, "result": np.nan}
    else:
        results = {"init": best_initial_value, "betas": best_betas, "result": best_result}

    results_df = pd.DataFrame(results, index=[0])
    results_df.to_csv(file_name)


def try_params(timestamps, fun, test_name, key):
    params_set, string_key = key
    file_name = "./data/{}_Params_{}.csv".format(test_name, string_key)
    best_initial_value, best_betas, best_result = -np.inf, -np.inf, -np.inf
    if timestamps:
        for params in params_set:
            try:
                optimised = fun(params, timestamps)
                result = -optimised.fun
                betas = optimised.x
                if result > best_result:
                    best_betas = betas
                    best_initial_value = params
                    best_result = result
                    if result - best_result < 1e-8:
                        write_results_(best_initial_value, best_betas, best_result, file_name)
                        return
            except:
                print("Iteration {} erred".format(params))
    write_results_(best_initial_value, best_betas, best_result, file_name)


def run_test(params, run_f, test_name, parallel=True):
    if test_name:
        _clean_directory(test_name)
    if parallel:
        p = Pool(processes=multiprocessing.cpu_count())
        result = p.map(run_f, params)
        p.close()
        p.join()
    else:
        return [run_f(p) for p in params]


def split_simple_config(params_space):
    return np.array_split(params_space, multiprocessing.cpu_count())


def split_complex_config(params_space):
    permutations = list(itertools.product(*params_space))
    return np.array_split(permutations, multiprocessing.cpu_count())
