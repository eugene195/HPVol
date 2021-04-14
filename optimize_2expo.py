from functools import partial
import numpy as np

from multiprocessing_sim import try_params, split_complex_config, run_test
from optimize_params import sum_n_exp_minimiser, get_timestamps


def test_2(timestamps):
    start = 0.001
    stop = 0.12
    N = 20
    test_name = "2Expo"
    test1_params = np.linspace(start, stop, int(N))
    param_configs = [(c, "{}".format(c[0])) for c in split_complex_config([test1_params] * 2)]
    run_simulation_func = partial(try_params, timestamps, sum_n_exp_minimiser, test_name)
    run_test(param_configs, run_simulation_func, test_name=test_name, parallel=True)


test_2(
    get_timestamps()
)