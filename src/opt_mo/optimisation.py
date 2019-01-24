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
    opponents,
    method_params={"n_random_starts": 20, "n_calls": 40},
    tol=10 ** -5,
):
    """
    Approximates the best response memory one strategy using bayesian optimisation.
    """
    bounds = [(0, 0.9999) for _ in range(4)]
    objective = prepare_objective_optimisation(opponents=opponents)

    default_calls = method_params["n_calls"]

    while method_params["n_calls"] == default_calls or (
        find_repeat_in_history(history, tol=tol) >= float("inf")
        and not objective_is_converged(values, tol=tol)
    ):
        result = skopt.gp_minimize(
            func=objective,
            dimensions=bounds,
            acq_func="EI",
            random_state=0,
            **method_params
        )

        history, values = result.x_iters, result.func_vals
        print(
            find_repeat_in_history(history, tol=tol),
            objective_is_converged(values, tol=tol),
        )
        method_params["n_calls"] += 20

    return list(result.x), -result.fun


def objective_is_converged(values, percent=10 ** -1, tol=10 ** -2):
    N = len(values)
    last = int(N * percent)
    values = sorted(values, reverse=True)
    return max(values[-last:]) - min(values[-last:]) < tol


def find_repeat_in_history(history, tol=10 ** -5):
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
            history[-cycle_size:],
            history[-2 * cycle_size : -cycle_size],
            atol=tol,
        ):
            return len(history[-cycle_size:])
    return float("inf")


def find_evolutionary_best_response(
    opponents, best_response_function, tol=10 ** -5
):
    history = []
    current = np.array([1, 1, 1, 1])
    best_response, _ = best_response_function(opponents + [current])

    while np.allclose(
        current, best_response
    ) is False or opt_mo.find_repeat_in_history(history) >= float("inf"):

        current = best_response
        history.append(current)
        best_response, _ = best_response_function(
            opponents + [current], tol=tol
        )

    return best_response, history
