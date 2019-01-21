import functools

import axelrod as axl
import numpy as np

import opt_mo


def test_prepare_objective_optimisation():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    objective = opt_mo.prepare_objective_optimisation(opponents)

    assert type(objective) == functools.partial


def test_bayesian_mem_one():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    turns, repetitions = 200, 5
    axl.seed(0)
    x, theor = opt_mo.memory_one_best_response(
        opponents=opponents,
        turns=turns,
        repetitions=repetitions,
        method_params={"n_random_starts": 20, "n_calls": 40},
    )

    assert len(x) == 4
    assert np.isclose(theor, 3.0, atol=10 ** -2)
