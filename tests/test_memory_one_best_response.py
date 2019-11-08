import random

import axelrod as axl
import numpy as np

import opt_mo


def test_prepare_objective_optimisation():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    objective = opt_mo.memory_one_best_response.prepare_objective_optimisation(
        opponents
    )
    player = [random.random() for _ in range(4)]

    assert type(objective) == type(lambda x: x)
    assert objective(player) <= 0


def test_bayesian_mem_one():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    axl.seed(0)
    best_response = opt_mo.get_memory_one_best_response(
        opponents=opponents, n_random_starts=20, n_calls=40
    )

    assert len(best_response) == 4


def test_objective_is_converge():
    values = [
        2.9859256920,
        3.0269409991,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
        3.0850608205,
    ]

    assert opt_mo.objective_is_converged(values) == True
