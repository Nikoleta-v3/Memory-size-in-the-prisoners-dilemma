"""
This file contains methods used in optimising memory one strategies and
retrieving the best responses.
"""
from functools import partial

import numpy as np
import skopt

import opt_mo


def prepare_objective_optimisation(opponents):
    objective = partial(opt_mo.tournament_utility, opponents=opponents)
    return lambda x: -objective(x)


def get_memory_one_best_response(
    opponents,
    method_params={"n_random_starts": 40, "n_calls": 60},
    tol=10 ** -5,
):
    """
    Approximates the best response memory one strategy using bayesian optimisation.
    """
    bounds = [(0, 0.9999) for _ in range(4)]
    objective = prepare_objective_optimisation(opponents=opponents)

    default_calls = method_params["n_calls"]

    while method_params[
        "n_calls"
    ] == default_calls or not objective_is_converged(values, tol=tol):
        result = skopt.gp_minimize(
            func=objective,
            dimensions=bounds,
            acq_func="EI",
            random_state=0,
            **method_params
        )

        values = result.func_vals
        print(objective_is_converged(values, tol=tol))
        method_params["n_calls"] += 20

    return np.array(result.x)


def objective_is_converged(values, percent=10 ** -1, tol=10 ** -2):
    N = len(values)
    last = int(N * percent)
    values = np.minimum.accumulate(values, 0)
    return max(values[-last:]) - min(values[-last:]) < tol
