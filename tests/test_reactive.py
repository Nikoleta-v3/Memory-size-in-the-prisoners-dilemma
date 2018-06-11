"""
Tests reactive.py
"""

import unittest
import opt_mo

import sympy as sym
import numpy as np
import axelrod as axl


class TestMatrixExpressions(unittest.TestCase):

    def test_round_expressions(self):
        x = sym.symbols('x')

        matrix = sym.Matrix([[x * 10 ** -18, 1], [3 * x * 10 ** -6, 0]])
        matrix = opt_mo.round_matrix_expressions(matrix, 4, x)

        self.assertEqual(matrix, sym.Matrix([[0, 1], [0, 0]]))

class TestEliminatorMethod(unittest.TestCase):
    x, y = sym.symbols('x, y')

    f = x ** 2 + x * y + 2.0 * x + y - 1.0
    g = x ** 2 + 3.0 * x - y ** 2 + 2.0 * y - 1.0
    system = [f, g]

    def test_eliminator_method(self):
        roots_for_x = opt_mo.eliminator_method(system=self.system, variable=self.x,
                                               other_variable=self.y)
        self.assertEqual(roots_for_x, set([0.9999999999999998, 0.0]))

    def test_solve_system(self):
        other_roots = [0.9999999999999998, 0.0]
        roots_for_y = opt_mo.solve_system(system=self.system, variable=self.y,
                                          other_roots=other_roots, other_variable=self.x)
        self.assertEqual(roots_for_y, set([1]))

class TestReactiveSet(unittest.TestCase):

    def test_reactive_set_against_cooperator(self):
        opponent = [(1, 1, 1, 1)]
        solution_set = opt_mo.reactive_set(opponent)
        self.assertEqual(solution_set, set([0, 1]))

    def test_reactive_set_against_defector(self):
        opponent = [(0, 0, 0, 0)]
        solution_set = opt_mo.reactive_set(opponent)
        self.assertEqual(solution_set, set([0, 1]))

    def test_reactive_set_against_player(self):
        axl.seed(21)
        opponent = [np.random.random(4)]

        solution_set = opt_mo.reactive_set(opponent)
        self.assertEqual(solution_set, set([0, 0.4450664548206672, 0.31493358733410926, 1]))

class TestPlayerForNumericalExperiment(unittest.TestCase):

    def test_result_from_numerical_experiments(self):
        axl.seed(2933)
        opponents = [np.random.random(4) for _ in range(1)]

        solution_set =  opt_mo.reactive_set(opponents)
        solution = opt_mo.argmax(opponents, solution_set)

        self.assertEqual(solution_set, set([0, .13036776235395534, 0.4267968584197452, 1]))
        self.assertEqual(solution, (0, 0.4267968584197452, 2.6546912335393573))