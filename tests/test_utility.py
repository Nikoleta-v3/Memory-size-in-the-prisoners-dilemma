import unittest
import opt_mo

import numpy as np

class TestNumeratorFormulation(unittest.TestCase):

    qs = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0)]
    def test_mem_quadratic_numerator(self):
        self.assertTrue((opt_mo.mem_quadratic_numerator(self.qs[0]) ==
                        np.array([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])).all())
        self.assertTrue((opt_mo.mem_quadratic_numerator(self.qs[1]) ==
                         np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])).all())
        self.assertTrue((opt_mo.mem_quadratic_numerator(self.qs[2]) ==
                         np.array([[0, 0, 1, -5],
                                   [0, 0, 0, 3],
                                   [1, 0, 0, 0],
                                   [-5, 3, 0, 0]])).all())

    def test_mem_linear_numerator(self):
        self.assertTrue((opt_mo.mem_linear_numerator(self.qs[0]) ==
                         np.array([ 0, -1,  0,  0])).all())
        self.assertTrue((opt_mo.mem_linear_numerator(self.qs[1]) ==
                         np.array([ -5, 0,  3,  0])).all())
        self.assertTrue((opt_mo.mem_linear_numerator(self.qs[2]) ==
                         np.array([ -1, 0,  -1,  5])).all())

    def test_mem_constant_numerator(self):
        self.assertEqual(opt_mo.mem_constant_numerator(self.qs[0]), 1)
        self.assertEqual(opt_mo.mem_constant_numerator(self.qs[1]), 5)
        self.assertEqual(opt_mo.mem_constant_numerator(self.qs[2]), 1)


class TestDenominatorFormulation(unittest.TestCase):
    
    qs = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0)]

    def test_mem_quadratic_denominator(self):
        self.assertTrue((opt_mo.mem_quadratic_denominator(self.qs[0]) ==
                        np.array([[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])).all())
        self.assertTrue((opt_mo.mem_quadratic_denominator(self.qs[1]) ==
                         np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])).all())
        self.assertTrue((opt_mo.mem_quadratic_denominator(self.qs[2]) ==
                         np.array([[0, 0, 1, -2],
                                   [0, 0, 0, 1],
                                   [1, 0, 0, 0],
                                   [-2, 1, 0, 0]])).all())

    def test_mem_linear_denominator(self):
        self.assertTrue((opt_mo.mem_linear_denominator(self.qs[0]) ==
                         np.array([ 0, -1,  0,  1])).all())
        self.assertTrue((opt_mo.mem_linear_denominator(self.qs[1]) ==
                         np.array([ -1, 0,  1,  0])).all())
        self.assertTrue((opt_mo.mem_linear_denominator(self.qs[2]) ==
                         np.array([ -1, 0, -1,  2])).all())

    def test_mem_constant_numerator(self):
        self.assertEqual(opt_mo.mem_constant_denominator(self.qs[0]), 1)
        self.assertEqual(opt_mo.mem_constant_denominator(self.qs[1]), 1)
        self.assertEqual(opt_mo.mem_constant_denominator(self.qs[2]), 1)

class TestUtility(unittest.TestCase):

    def test_against_defector(self):
        q = (0, 0, 0, 0)
        self.assertEqual(opt_mo.utility((0, 0, 0, 0), q), 1)
        self.assertEqual(opt_mo.utility((1, 1, 1, 1), q), 0)

    def test_against_cooperator(self):
        q = (1, 1, 1, 1)
        self.assertEqual(opt_mo.utility((0, 0, 0, 0), q), 5)
        self.assertEqual(opt_mo.utility((1, 1, 1, 1), q), 3)

    def test_against_tit_for_tat(self):
        q = (1, 0, 1, 0)
        self.assertEqual(opt_mo.utility((0, 0, 0, 0), q), 1)
        self.assertEqual(opt_mo.utility((1, 1, 1, 1), q), 3)

class TestTournamentUtility(unittest.TestCase):
    def test_against_defectors(self):
        q = (0, 0, 0, 0)
        z = (0, 0, 0, 0)
        self.assertEqual(opt_mo.tournament_utility((0, 0, 0, 0), [q, z]), 1)
        self.assertEqual(opt_mo.tournament_utility((1, 1, 1, 1), [q, z]), 0)

    def test_against_cooperators(self):
        q = (1, 1, 1, 1)
        z = (1, 1, 1, 1)
        self.assertEqual(opt_mo.tournament_utility((0, 0, 0, 0), [q, z]), 5)
        self.assertEqual(opt_mo.tournament_utility((1, 1, 1, 1), [q, z]), 3)

    def test_against_cooperator_defector(self):
        q = (1, 1, 1, 1)
        z = (0, 0, 0, 0)
        self.assertEqual(opt_mo.tournament_utility((0, 0, 0, 0), [q, z]), 3)
        self.assertEqual(opt_mo.tournament_utility((1, 1, 1, 1), [q, z]), 1.5)