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
    return (
        lambda x: -objective(x)
        if (np.isnan(objective(x)) == False and np.isinf(objective(x)) == False)
        else 100
    )


def get_memory_one_best_response(
    opponents,
    n_random_starts=40,
    n_calls=60,
    tol=10 ** -5,
    convergence_switch=True,
):
    """
    Approximates the best response memory one strategy using bayesian optimisation.
    """
    bounds = [(0, 1.0) for _ in range(4)]
    objective = prepare_objective_optimisation(opponents=opponents)

    method_params = {"n_random_starts": n_random_starts, "n_calls": n_calls}
    default_calls = n_calls

    while (
        method_params["n_calls"] == default_calls
        or not objective_is_converged(values, tol=tol)
    ) and convergence_switch:
        result = skopt.gp_minimize(
            func=objective,
            dimensions=bounds,
            acq_func="EI",
            random_state=0,
            **method_params
        )

        values = result.func_vals
        method_params["n_calls"] += 20

    return np.array(result.x)


def objective_is_converged(values, percent=10 ** -1, tol=10 ** -2):
    N = len(values)
    last = int(N * percent)
    values = np.minimum.accumulate(values, 0)
    return max(values[-last:]) - min(values[-last:]) < tol
