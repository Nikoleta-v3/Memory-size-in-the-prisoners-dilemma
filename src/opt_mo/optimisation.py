"""
This file contains methods used in optimising memory one strategies and
retrieving the best responses.
"""
from functools import partial

import skopt

import opt_mo


def prepare_objective_optimisation(opponents):
    objective = partial(opt_mo.tournament_utility, opponents=opponents)
    return objective


def memory_one_best_response(opponents, turns, repetitions, method_params):
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

    return list(result.x)
