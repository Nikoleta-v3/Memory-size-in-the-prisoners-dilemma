"""
Tests reactive.py
"""
import axelrod as axl
import numpy as np
import sympy as sym

import opt_mo


def test_round_expressions():
    x = sym.symbols("x")

    matrix = sym.Matrix([[x * 10 ** -18, 1], [3 * x * 10 ** -6, 0]])
    matrix = opt_mo.reactive.round_matrix_expressions(matrix, 4, x)

    assert matrix, sym.Matrix([[0, 1], [0, 0]])


x, y = sym.symbols("x, y")

f = x ** 2 + x * y + 2.0 * x + y - 1.0
g = x ** 2 + 3.0 * x - y ** 2 + 2.0 * y - 1.0
system = [f, g]


def test_eliminator_method():
    roots_for_x = opt_mo.reactive.eliminator_method(
        system=system, variable=x, other_variable=y
    )

    assert roots_for_x == set([0.9999999999999998, 0.0])


def test_solve_system():
    other_roots = [0.9999999999999998, 0.0]
    roots_for_y = opt_mo.reactive.solve_system(
        system=system, variable=y, other_roots=other_roots, other_variable=x
    )
    assert roots_for_y == set([1])


def test_reactive_set_against_player():
    axl.seed(21)
    opponent = [np.random.random(4)]

    solution_set = opt_mo.reactive.reactive_set(opponent)
    assert solution_set == set([0, 0.4450664548206673, 0.31493358733410926, 1])


def test_result_from_numerical_experiments():
    axl.seed(2933)
    opponents = [np.random.random(4) for _ in range(1)]

    solution_set = opt_mo.reactive.reactive_set(opponents)
    solution = opt_mo.reactive.argmax(opponents, solution_set)

    assert solution_set == set([0, .13036776235395545, 0.4267968584197451, 1])
    assert solution == (0, 0.4267968584197451, 2.6546912335393573)


def test_reactive_best_response_against_cooperator():
    opponent = [(1, 1, 1, 1)]
    best_response = opt_mo.reactive_best_response(opponent)

    assert np.array_equal(best_response, [0, 0, 0, 0])


def test_reactive_best_response_against_defector():
    opponent = [(0, 0, 0, 0)]
    best_response = opt_mo.reactive_best_response(opponent)

    assert np.array_equal(best_response, [0, 0, 0, 0])
