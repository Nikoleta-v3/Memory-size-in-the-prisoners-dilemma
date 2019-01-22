import functools

import axelrod as axl
import numpy as np

import opt_mo


def test_prepare_objective_optimisation():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    objective = opt_mo.optimisation.prepare_objective_optimisation(opponents)

    assert type(objective) == functools.partial


def test_bayesian_mem_one():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    axl.seed(0)
    x, theor = opt_mo.memory_one_best_response(
        opponents=opponents,
        method_params={"n_random_starts": 20, "n_calls": 40},
    )

    assert len(x) == 4
    assert np.isclose(theor, 3.0, atol=10 ** -2)


def test_find_evolutionary_best_response():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    opt_player, hist = opt_mo.find_evolutionary_best_response(
        opponents, opt_mo.memory_one_best_response
    )

    expected = [0.3647357415723788, 0.030269685380524037, 0.0, 0.0]
    assert len(hist) == 2
    assert np.allclose(opt_player, expected)
