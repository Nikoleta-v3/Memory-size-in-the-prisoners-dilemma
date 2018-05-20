import imp
main = imp.load_source('main', '../main.py')

import os
import unittest
import numpy as np
import axelrod as axl
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class TestGetFilename(unittest.TestCase):

    def test_example_one(self):
        location = '/home/Documents/'
        folder = 'matches'
        params = [0, 1, 1]
        index=0

        filename = main.get_filename(location, folder, params, index)
        self.assertEqual(filename, '/home/Documents/gambler0_1_1/matches/0.csv')

    def test_example_two(self):
        location = ''
        folder = 'tournaments'
        params = [2, 1, 1]
        index=1

        filename = main.get_filename(location, folder, params, index)
        self.assertEqual(filename, 'gambler2_1_1/tournaments/1.csv')

class TestObjectiveScore(unittest.TestCase):
    turns, repetitions, params = 100, 5, [1, 1, 1]

    def test_against_defector(self):
        opponent = [(0, 0, 0, 0)]
        pattern_1 = [0 for _ in range(9)]
        pattern_2 = [1 for _ in range(9)]

        obj_1 = main.objective_score(pattern=pattern_1, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        obj_2 = main.objective_score(pattern=pattern_2, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        self.assertTrue(np.isclose(abs(obj_1), 1.04))
        self.assertTrue(np.isclose(abs(obj_2), 0.03))

    def test_against_cooperator(self):
        opponent = [(1, 1, 1, 1)]
        pattern_1 = [0 for _ in range(9)]
        pattern_2 = [1 for _ in range(9)]

        obj_1 = main.objective_score(pattern=pattern_1, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        obj_2 = main.objective_score(pattern=pattern_2, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        self.assertTrue(np.isclose(abs(obj_1), 5))
        self.assertTrue(np.isclose(abs(obj_2), 3))

    def test_against_tit_for_tat(self):
        opponent = [(1, 0, 1, 0)]
        pattern_1 = [0 for _ in range(9)]
        pattern_2 = [1 for _ in range(9)]

        obj_1 = main.objective_score(pattern=pattern_1, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        obj_2 = main.objective_score(pattern=pattern_2, turns=self.turns,
                                     repetitions=self.repetitions, opponents=opponent,
                                     params=self.params)
        self.assertTrue(np.isclose(abs(obj_1), 1.04))
        self.assertTrue(np.isclose(abs(obj_2), 3))

class TestPatternSize(unittest.TestCase):

    def test_example_one(self):
        params = [1, 1, 1]
        self.assertEqual(main.pattern_size(params), 8)

    def test_example_two(self):
        params = [1, 1, 2]
        self.assertEqual(main.pattern_size(params), 16)

class TestTrainGambler(unittest.TestCase):
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    turns, repetitions = 10, 2

    def test_bayesian(self):
        axl.seed(0)
        x, fun = main.train_gambler(method='bayesian', opponents=self.opponents,
                                    turns=self.turns, repetitions=self.repetitions,
                                    params=[1, 1, 2], n_random_starts=10)

        self.assertEqual(len(x), main.pattern_size([1, 1, 2]) + 1)
        self.assertTrue(np.isclose(fun, 2.9, atol=10 ** -1))

    def test_differential(self):
        axl.seed(0)
        x, fun = main.train_gambler(method='differential', opponents=self.opponents,
                                    turns=self.turns, repetitions=self.repetitions,
                                    params=[0, 0, 1], popsize=5)

        self.assertEqual(len(x), main.pattern_size([0, 0, 1]) + 1)
        self.assertTrue(np.isclose(fun, 3.2, atol=10 ** -1))

class TestOptimalMemoryOne(unittest.TestCase):
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    turns, repetitions = 200, 5

    def test_bayesian(self):
        axl.seed(0)
        x, theor, simul = main.optimal_memory_one(method='differential', opponents=self.opponents,
                                                  turns=self.turns, repetitions=self.repetitions)

        self.assertEqual(len(x), 4)
        self.assertTrue(np.isclose(theor, 3.0, atol=10 ** -2))
        self.assertTrue(np.isclose(simul, theor, atol=10 ** -2))

    def test_differential(self):
        axl.seed(0)
        x, theor, simul = main.optimal_memory_one(method='bayesian', opponents=self.opponents,
                                                  turns=self.turns, repetitions=self.repetitions,
                                                  n_calls=40)

        self.assertEqual(len(x), 4)
        self.assertTrue(np.isclose(theor, 3.0, atol=10 ** -2))
        self.assertTrue(np.isclose(simul, theor, atol=10 ** -1))

# class TestWriteFile(unittest.TestCase):
#     method='bayesian'
#     params=[0, 0, 1]
#     turns=10
#     repetitions=2

#     def test_match(self):
#         filename = 'match_example.csv'
#         main.write_results(method=self.method, list_opponents=[[0, 0, 0, 0]],
#                            filename=filename, params=self.params,
#                            turns=self.turns, repetitions=self.repetitions)
#         df = pd.read_csv(filename)

#         self.assertEqual(len(df.columns), 18)
#         self.assertTrue(df[r'$\bar{q}_1$'].isnull().all())
#         self.assertTrue(df[r'$\bar{q}_2$'].isnull().all())
#         self.assertTrue(df[r'$\bar{q}_3$'].isnull().all())
#         self.assertTrue(df[r'$\bar{q}_4$'].isnull().all())

#         os.remove(filename)

#     def test_tournament(self):
#         filename = 'tournament_example.csv'
#         main.write_results(method=self.method, list_opponents=[[0, 0, 0, 0], [1, 1, 1, 1]],
#                            filename=filename, params=self.params,
#                            turns=self.turns, repetitions=self.repetitions)
#         df = pd.read_csv(filename)

#         self.assertEqual(len(df.columns), 18)
#         self.assertFalse(df[r'$\bar{q}_1$'].isnull().all())
#         self.assertFalse(df[r'$\bar{q}_2$'].isnull().all())
#         self.assertFalse(df[r'$\bar{q}_3$'].isnull().all())
#         self.assertFalse(df[r'$\bar{q}_4$'].isnull().all())

#         os.remove(filename)

class TestGetColumns(unittest.TestCase):
    # without gambler
    len_cols = 22

    def test_get_columns_no_params(self):
        params = [0, 0, 0]
        self.assertEqual(len(main.get_columns(params)), self.len_cols + 1)

    def test_get_columns_op_initial_move(self):
        params = [0, 0, 1]
        self.assertEqual(len(main.get_columns(params)), self.len_cols + 2)

    def test_get_columns(self):
        params = [0, 1, 1]
        self.assertEqual(len(main.get_columns(params)), self.len_cols + 4)