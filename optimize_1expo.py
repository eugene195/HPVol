from functools import partial
import numpy as np

from multiprocessing_sim import split_simple_config, try_params, run_test
from optimize_params import exp_minimiser, get_timestamps


def test_1(timestamps):
    start = 0.001
    stop = 1.25
    N = 1e6
    test_name = "1Expo"
    test1_params = np.linspace(start, stop, int(N))
    param_configs = [(c, "{}".format(c[0])) for c in split_simple_config(test1_params)]
    run_simulation_func = partial(try_params, timestamps, exp_minimiser, test_name)
    run_test(param_configs, run_simulation_func, test_name=test_name, parallel=True)

test_1(
    get_timestamps()
)
