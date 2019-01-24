import functools

import axelrod as axl
import numpy as np

import opt_mo


def test_prepare_objective_optimisation():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    objective = opt_mo.optimisation.prepare_objective_optimisation(opponents)

    assert type(objective) == functools.partial


def test_bayesian_mem_one():
    opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
    axl.seed(0)
    x, theor = opt_mo.memory_one_best_response(
        opponents=opponents,
        method_params={"n_random_starts": 20, "n_calls": 40},
    )

    assert len(x) == 4
    assert np.isclose(theor, 3.0, atol=10 ** -2)


# def test_find_evolutionary_best_response():
#     opponents = [[0, 0, 0, 0], [1, 1, 1, 1]]
#     opt_player, hist = opt_mo.find_evolutionary_best_response(
#         opponents, opt_mo.memory_one_best_response
#     )

#     expected = [0.3647357415723788, 0.030269685380524037, 0.0, 0.0]
#     assert len(hist) == 2
#     assert np.allclose(opt_player, expected)


def test_find_repeat_in_history_with_single_element():
    history_with_single_element_cycle = (
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .4, .3, .3]),
    )

    assert np.array_equal(
        opt_mo.find_repeat_in_history(history_with_single_element_cycle),
        (np.array([0.3, 0.4, 0.3, 0.3]),),
    )


def test_find_repeat_in_history_with_two_elements():
    history_with_two_element_cycle = (
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
    )

    assert np.array_equal(
        opt_mo.find_repeat_in_history(history_with_two_element_cycle),
        (np.array([0.3, 0.4, 0.3, 0.3]), np.array([0.3, 0.4, 0.3, 0.4])),
    )


def test_find_repeat_in_history_with_three_elements():
    history_with_three_element_cycle = (
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.7, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.3, .4, .2, .4]),
        np.array([.7, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.3, .4, .2, .4]),
    )

    assert np.array_equal(
        opt_mo.find_repeat_in_history(history_with_three_element_cycle),
        (
            np.array([0.7, 0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3, 0.4]),
            np.array([0.3, 0.4, 0.2, 0.4]),
        ),
    )


def test_find_repeat_in_history_with_three_elements_and_floating_error():
    history_with_three_element_cycle_and_floating_point_error = (
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.7, .4, .3, .3]),
        np.array([.3 + 10 ** -10, .4, .3, .4]),
        np.array([.3, .4, .2, .4 + 10 ** -10]),
        np.array([.7, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.3, .4, .2, .4]),
    )

    assert np.array_equal(
        opt_mo.find_repeat_in_history(
            history_with_three_element_cycle_and_floating_point_error
        ),
        (
            np.array([0.7, 0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3, 0.4]),
            np.array([0.3, 0.4, 0.2, 0.4]),
        ),
    )


def test_find_repeat_in_history_no_cycle():
    history_with_no_cycle = (
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.7, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
        np.array([.3, .4, .2, .4]),
        np.array([.7, .4, .3, .3]),
        np.array([.3, .4, .3, .4]),
    )

    assert opt_mo.find_repeat_in_history(history_with_no_cycle) is False


def test_find_repeat_cycle_from_the_beginning():
    history_with_cycle_from_the_beginning = (
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
        np.array([.3, .4, .3, .3]),
        np.array([.3, .7, .3, .8]),
    )

    assert np.array_equal(
        opt_mo.find_repeat_in_history(history_with_cycle_from_the_beginning),
        (np.array([.3, .4, .3, .3]), np.array([.3, .7, .3, .8])),
    )
