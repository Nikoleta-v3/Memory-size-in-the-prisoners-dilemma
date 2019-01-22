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
    return objective


def memory_one_best_response(
    opponents, method_params={"n_random_starts": 20, "n_calls": 40}
):
    """
    Approximates the best response memory one strategy using bayesian optimisation.
    """
    bounds = [(0, 0.9999) for _ in range(4)]
    objective = prepare_objective_optimisation(opponents=opponents)

    result = skopt.gp_minimize(
        func=objective,
        dimensions=bounds,
        acq_func="EI",
        random_state=0,
        **method_params
    )

    return list(result.x), -result.fun


def find_evolutionary_best_response(opponents, best_response_function):
    history = []
    current = np.array([1, 1, 1, 1])

    best_response, _ = best_response_function(opponents + [current])

    while np.allclose(current, best_response) is False:
        current = best_response
        history.append(current)
        best_response, _ = best_response_function(opponents + [current])
    return best_response, history
