"""
Tests reactive.py
"""
import random

import numpy as np
import sympy as sym

import axelrod as axl
import opt_mo

x, y = sym.symbols("x, y")

f = x ** 2 + x * y + 2.0 * x + y - 1.0
g = x ** 2 + 3.0 * x - y ** 2 + 2.0 * y - 1.0
system = [f, g]


def test_roots_using_eliminator_method():
    roots_for_x = opt_mo.reactive_best_response._roots_using_eliminator_method(
        system=system, variable=x, other_variable=y
    )

    assert roots_for_x == set([0.9999999999999998, 0.0])


def test_solve_system():
    other_roots = [0.9999999999999998, 0.0]
    roots_for_y = opt_mo.reactive_best_response._roots_solving_system_of_singel_unknown(
        system=system, variable=y, other_roots=other_roots, other_variable=x
    )
    assert roots_for_y == set([1])


def test_reactive_set_against_player():
    axl.seed(21)
    opponent = [np.random.random(4)]

    solution_set = opt_mo.reactive_best_response.get_candinate_reactive_best_responses(
        opponent
    )
    assert solution_set == set([0, 0.4450664548206673, 0.31493358733410926, 1])


def test_result_from_numerical_experiments():
    axl.seed(2933)
    opponents = [np.random.random(4) for _ in range(1)]

    solution_set = opt_mo.reactive_best_response.get_candinate_reactive_best_responses(
        opponents
    )
    solution = opt_mo.reactive_best_response.get_argmax(opponents, solution_set)

    assert len(solution_set) == len(
        set([0, 0.13036776235395545, 0.4267968584197451, 1])
    )
    assert np.allclose(solution, (0, 0.4267968584197451, 2.6546912335393573))


def test_reactive_best_response_against_cooperator():
    opponent = [(1, 1, 1, 1)]
    best_response = opt_mo.get_reactive_best_response(opponent)

    assert np.array_equal(best_response, [0, 0, 0, 0])


def test_reactive_best_response_against_defector():
    opponent = [(0, 0, 0, 0)]
    best_response = opt_mo.get_reactive_best_response(opponent)

    assert np.array_equal(best_response, [0, 0, 0, 0])


def test_prepare_reactive_objective_optimisation():
    players = [[random.random() for _ in range(4)] for _ in range(3)]
    objective = opt_mo.memory_one_best_response.prepare_objective_optimisation(
        opponents=players[:-1]
    )

    assert type(objective) == type(lambda x: x)
    assert objective(players[-1]) <= 0


def test_reactive_best_response_bayesian():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    axl.seed(0)
    best_response = opt_mo.get_reactive_best_response_with_bayesian(
        opponents=opponents, n_random_starts=20, n_calls=40
    )

    assert len(best_response) == 4
    assert best_response[0] == best_response[2]
    assert best_response[1] == best_response[-1]
