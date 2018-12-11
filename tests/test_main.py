import imp
main = imp.load_source('main', '../main.py')

import os
import unittest
import numpy as np
import pandas as pd

class TestGetFilename(unittest.TestCase):

    def test_example_one(self):
        location = '/home/Documents/'
        params = [0, 1, 1]
        index = 0
        method = 'bayesian'

        filename = main.get_filename(location, params, index, method)
        self.assertEqual(filename, '/home/Documents/gambler0_1_1/bayesian_0.csv')

    def test_example_two(self):
        location = ''
        params = [2, 1, 1]
        index = 1
        method = 'differential'

        filename = main.get_filename(location, params, index, method)
        self.assertEqual(filename, 'gambler2_1_1/differential_1.csv')

class TestWriteFile(unittest.TestCase):
    method = 'bayesian'
    params = [0, 0, 1]
    turns = 10
    repetitions = 2
    method_params = {'n_random_starts' : 1, 'n_calls': 5}

    def test_match(self):
        # filename = 'match_example.csv'
        df = main.write_results(index=1, method=self.method, list_opponents=[[0, 0, 0, 0]],
                                params=self.params, turns=self.turns,
                                repetitions=self.repetitions, method_params=self.method_params)

        self.assertEqual(len(df.columns), 26)
        self.assertEqual(df['turns'].values[0], 10)
        self.assertEqual(df['repetitions'].values[0], 2)
        self.assertTrue(df[r'$\bar{q}_1$'].isnull().all())
        self.assertTrue(df[r'$\bar{q}_2$'].isnull().all())
        self.assertTrue(df[r'$\bar{q}_3$'].isnull().all())
        self.assertTrue(df[r'$\bar{q}_4$'].isnull().all())

    def test_tournament(self):
        # filename = 'tournament_example.csv'
        df = main.write_results(index=1, method=self.method, list_opponents=[[0, 0, 0, 0],
                                [1, 1, 1, 1]], params=self.params, turns=self.turns,
                                repetitions=self.repetitions, method_params=self.method_params)

        self.assertEqual(len(df.columns), 26)
        self.assertFalse(df[r'$\bar{q}_1$'].isnull().all())
        self.assertFalse(df[r'$\bar{q}_2$'].isnull().all())
        self.assertFalse(df[r'$\bar{q}_3$'].isnull().all())
        self.assertFalse(df[r'$\bar{q}_4$'].isnull().all())

class TestGetColumns(unittest.TestCase):
    # without gambler
    len_cols = 24

    def test_get_columns_no_params(self):
        params = [0, 0, 0]
        method_params = {'n_random_starts' : 1, 'n_calls': 1}
        self.assertEqual(len(main.get_columns(params, method_params)), self.len_cols + 1)

    def test_get_columns_op_initial_move(self):
        params = [0, 0, 1]
        method_params = {'n_random_starts' : 1, 'n_calls': 1}
        self.assertEqual(len(main.get_columns(params, method_params)), self.len_cols + 2)

    def test_get_columns(self):
        params = [0, 1, 1]
        method_params = {'n_random_starts' : 1, 'n_calls': 1}
        self.assertEqual(len(main.get_columns(params, method_params)), self.len_cols + 4)