import unittest
import opt_mo

import numpy as np
import axelrod as axl

class TestObjectiveScore(unittest.TestCase):
    turns, repetitions, params = 100, 5, [1, 1, 1]

    def test_against_defector(self):
        opponent = [(0, 0, 0, 0)]
        pattern_1 = [0 for _ in range(9)]
        pattern_2 = [1 for _ in range(9)]

        obj_1 = opt_mo.objective_score(pattern=pattern_1, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        obj_2 = opt_mo.objective_score(pattern=pattern_2, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        self.assertTrue(np.isclose(abs(obj_1), 1.04))
        self.assertTrue(np.isclose(abs(obj_2), 0.03))

    def test_against_cooperator(self):
        opponent = [(1, 1, 1, 1)]
        pattern_1 = [0 for _ in range(9)]
        pattern_2 = [1 for _ in range(9)]

        obj_1 = opt_mo.objective_score(pattern=pattern_1, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        obj_2 = opt_mo.objective_score(pattern=pattern_2, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        self.assertTrue(np.isclose(abs(obj_1), 5))
        self.assertTrue(np.isclose(abs(obj_2), 3))

    def test_against_tit_for_tat(self):
        opponent = [(1, 0, 1, 0)]
        pattern_1 = [0 for _ in range(9)]
        pattern_2 = [1 for _ in range(9)]

        obj_1 = opt_mo.objective_score(pattern=pattern_1, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        obj_2 = opt_mo.objective_score(pattern=pattern_2, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        self.assertTrue(np.isclose(abs(obj_1), 1.04))
        self.assertTrue(np.isclose(abs(obj_2), 3))

class TestPatternSize(unittest.TestCase):

    def test_example_one(self):
        params = [1, 1, 1]
        self.assertEqual(opt_mo.pattern_size(params), 8)

    def test_example_two(self):
        params = [1, 1, 2]
        self.assertEqual(opt_mo.pattern_size(params), 16)

class TestTrainGambler(unittest.TestCase):
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    turns, repetitions = 10, 2

    def test_bayesian(self):
        axl.seed(0)
        x, fun = opt_mo.train_gambler(method='bayesian', opponents=self.opponents,
                                    turns=self.turns, repetitions=self.repetitions,
                                    params=[1, 1, 2],
                                    method_params ={'n_random_starts' : 10,
                                                    'n_calls': 15})

        self.assertEqual(len(x), opt_mo.pattern_size([1, 1, 2]) + 1)
        self.assertTrue(np.isclose(fun, 2.9, atol=10 ** -1))

    def test_differential(self):
        axl.seed(0)
        x, fun = opt_mo.train_gambler(method='differential', opponents=self.opponents,
                                    turns=self.turns, repetitions=self.repetitions,
                                    params=[0, 0, 1], method_params={'popsize': 5})

        self.assertEqual(len(x), opt_mo.pattern_size([0, 0, 1]) + 1)
        self.assertTrue(np.isclose(fun, 3.2, atol=10 ** -1))

class TestOptimalMemoryOne(unittest.TestCase):
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    turns, repetitions = 200, 5

    def test_bayesian(self):
        axl.seed(0)
        x, theor, simul = opt_mo.optimal_memory_one(method='differential', opponents=self.opponents,
                                                  turns=self.turns, repetitions=self.repetitions,
                                                  method_params={'popsize': 100})

        self.assertEqual(len(x), 4)
        self.assertTrue(np.isclose(theor, 3.0, atol=10 ** -2))
        self.assertTrue(np.isclose(simul, theor, atol=10 ** -2))

    def test_differential(self):
        axl.seed(0)
        x, theor, simul = opt_mo.optimal_memory_one(method='bayesian', opponents=self.opponents,
                                                  turns=self.turns, repetitions=self.repetitions,
                                                  method_params ={'n_random_starts' : 20,
                                                                  'n_calls': 40})

        self.assertEqual(len(x), 4)
        self.assertTrue(np.isclose(theor, 3.0, atol=10 ** -2))
        self.assertTrue(np.isclose(simul, theor, atol=10 ** -1))