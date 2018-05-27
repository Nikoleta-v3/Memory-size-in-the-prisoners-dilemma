"""
Tests reactive.py
"""

import unittest
import opt_mo

import sympy as sym

class TestMatrixExpressions(unittest.TestCase):

    def test_round_expressions(self):
        x = sym.symbols('x')

        matrix = sym.Matrix([[x * 10 ** -18, 1], [3 * x * 10 ** -6, 0]])
        matrix = opt_mo.round_matrix_expressions(matrix, 4, x)

        self.assertEqual(matrix, sym.Matrix([[0, 1], [0, 0]]))

class TestEliminatorMethod(unittest.TestCase):

    def test_get_roots_eliminator_method(self):
        x, y = sym.symbols('x, y')

        f = x ** 2 + x * y + 2.0 * x + y -1.0
        g = x ** 2 + 3.0 * x - y ** 2 + 2.0 * y - 1.0
        system = [f, g]

        roots_for_x = opt_mo.get_roots_eliminator_method(system, x, y)
        self.assertEqual(roots_for_x, [0.9999999999999998,0.0])