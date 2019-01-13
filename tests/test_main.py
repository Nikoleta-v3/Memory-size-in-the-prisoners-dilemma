import imp
import os

import numpy as np
import pandas as pd

from opt_mo import main


def test_get_filename_example_one():
    location = "/home/Documents/"
    params = [0, 1, 1]
    index = 0

    filename = main.get_filename(location, params, index)
    assert filename == "/home/Documents/gambler0_1_1/0.csv"


def test_write_file_match():
    params = [0, 0, 1]
    turns = 10
    repetitions = 2
    method_params = {"n_random_starts": 1, "n_calls": 5}

    df = main.write_results(
        index=1,
        list_opponents=[[0, 0, 0, 0]],
        params=params,
        turns=turns,
        repetitions=repetitions,
        method_params=method_params,
    )

    assert len(df.columns) == 25
    assert df["turns"].values[0] == 10
    assert df["repetitions"].values[0] == 2
    assert df[r"$\bar{q}_1$"].isnull().all()
    assert df[r"$\bar{q}_2$"].isnull().all()
    assert df[r"$\bar{q}_3$"].isnull().all()
    assert df[r"$\bar{q}_4$"].isnull().all()


def test_write_tournament():
    params = [0, 0, 1]
    turns = 10
    repetitions = 2
    method_params = {"n_random_starts": 1, "n_calls": 5}
    df = main.write_results(
        index=1,
        list_opponents=[[0, 0, 0, 0], [1, 1, 1, 1]],
        params=params,
        turns=turns,
        repetitions=repetitions,
        method_params=method_params,
    )

    assert len(df.columns) == 25
    assert df[r"$\bar{q}_1$"].isnull().all() == False
    assert df[r"$\bar{q}_2$"].isnull().all() == False
    assert df[r"$\bar{q}_3$"].isnull().all() == False
    assert df[r"$\bar{q}_4$"].isnull().all() == False


def test_get_columns_no_params():
    len_cols = 23
    params = [0, 0, 0]
    method_params = {"n_random_starts": 1, "n_calls": 1}
    assert len(main.get_columns(params, method_params)) == len_cols + 1


def test_get_columns_op_initial_move():
    len_cols = 23
    params = [0, 0, 1]
    method_params = {"n_random_starts": 1, "n_calls": 1}
    assert len(main.get_columns(params, method_params)) == len_cols + 2


def test_get_columns():
    len_cols = 23
    params = [0, 1, 1]
    method_params = {"n_random_starts": 1, "n_calls": 1}
    assert len(main.get_columns(params, method_params)) == len_cols + 4
