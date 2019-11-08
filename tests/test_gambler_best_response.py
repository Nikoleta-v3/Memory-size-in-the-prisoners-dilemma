import axelrod as axl
import numpy as np

import opt_mo


def test_params_size_example_one():
    params = [1, 1, 1]
    assert opt_mo.gambler_best_response.get_lookup_table_size(params) == 8


def test_params_size_example_two():
    params = [1, 1, 2]
    assert opt_mo.gambler_best_response.get_lookup_table_size(params) == 16


def test_tournament_score_gambler_against_defector():
    opponent = [(0, 0, 0, 0)]
    pattern_1 = [0 for _ in range(9)]
    pattern_2 = [1 for _ in range(9)]
    turns, repetitions, params = 100, 5, [1, 1, 1]

    obj_1 = opt_mo.tournament_score_gambler(
        pattern=pattern_1,
        turns=turns,
        repetitions=repetitions,
        opponents=opponent,
        params=params,
    )
    obj_2 = opt_mo.tournament_score_gambler(
        pattern=pattern_2,
        turns=turns,
        repetitions=repetitions,
        opponents=opponent,
        params=params,
    )
    assert np.isclose(abs(obj_1), 1.04)
    assert np.isclose(abs(obj_2), 0.03)


def test_tournament_score_gambler_against_cooperator():
    opponent = [(1, 1, 1, 1)]
    pattern_1 = [0 for _ in range(9)]
    pattern_2 = [1 for _ in range(9)]
    turns, repetitions, params = 100, 5, [1, 1, 1]

    obj_1 = opt_mo.tournament_score_gambler(
        pattern=pattern_1,
        turns=turns,
        repetitions=repetitions,
        opponents=opponent,
        params=params,
    )
    obj_2 = opt_mo.tournament_score_gambler(
        pattern=pattern_2,
        turns=turns,
        repetitions=repetitions,
        opponents=opponent,
        params=params,
    )
    assert np.isclose(abs(obj_1), 5)
    assert np.isclose(abs(obj_2), 3)


def test_tournament_score_gambler_against_tit_for_tat():
    opponent = [(1, 0, 1, 0)]
    pattern_1 = [0 for _ in range(9)]
    pattern_2 = [1 for _ in range(9)]
    turns, repetitions, params = 100, 5, [1, 1, 1]

    obj_1 = opt_mo.tournament_score_gambler(
        pattern=pattern_1,
        turns=turns,
        repetitions=repetitions,
        opponents=opponent,
        params=params,
    )
    obj_2 = opt_mo.tournament_score_gambler(
        pattern=pattern_2,
        turns=turns,
        repetitions=repetitions,
        opponents=opponent,
        params=params,
    )
    assert np.isclose(abs(obj_1), 1.04)
    assert np.isclose(abs(obj_2), 3)


def test_train_gambler():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    turns, repetitions = 10, 2
    axl.seed(0)
    x, score = opt_mo.get_best_response_gambler(
        opponents=opponents,
        turns=turns,
        repetitions=repetitions,
        params=[1, 1, 2],
        method_params={"n_random_starts": 10, "n_calls": 15},
    )

    assert (
        len(x)
        == opt_mo.gambler_best_response.get_lookup_table_size([1, 1, 2]) + 1
    )
