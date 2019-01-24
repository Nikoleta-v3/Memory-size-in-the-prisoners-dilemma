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


def find_repeat_in_history(history):
    """
    Find any repeat of last moves in history. Including repeats of just one element 
    (so this can be used to check simple convergence as well as "cyclic convergence").

    Parameters
    ==========

     history: any iterator of numpy arrays.
    """
    size = len(history)
    for cycle_size in range(1, int(size / 2) + 1):
        if np.allclose(
            history[-cycle_size:], history[-2 * cycle_size : -cycle_size]
        ):
            return len(history[-cycle_size:])
    return float('inf')


def find_evolutionary_best_response(opponents, best_response_function):
    history = []
    current, _ = best_response_function(opponents)
    cycle = True
    best_response, _ = best_response_function(opponents + [current])

    while np.allclose(current, best_response) is False:
        current = best_response
        history.append(current)
        iterations += 1
        if iterations % 10 == 0:
            print("Note history is at: {} iterations.".format(iterations))
        best_response, _ = best_response_function(opponents + [current])
    return best_response, history
