import random

import axelrod as axl
import numpy as np

import opt_mo


def test_get_repeat_length_in_history_with_single_element():
    history_with_single_element_cycle = (
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.3]),
    )

    assert (
        opt_mo.get_repeat_length_in_history(history_with_single_element_cycle)
        == 1
    )


def test_get_repeat_length_in_history_with_two_elements():
    history_with_two_element_cycle = (
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
    )

    assert (
        opt_mo.get_repeat_length_in_history(history_with_two_element_cycle) == 2
    )


def test_get_repeat_length_in_history_with_three_elements():
    history_with_three_element_cycle = (
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.7, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.2, 0.4]),
        np.array([0.7, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.2, 0.4]),
    )

    assert (
        opt_mo.get_repeat_length_in_history(history_with_three_element_cycle)
        == 3
    )


def test_get_repeat_length_in_history_with_three_elements_and_floating_error():
    history_with_three_element_cycle_and_floating_point_error = (
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.7, 0.4, 0.3, 0.3]),
        np.array([0.3 + 10 ** -10, 0.4, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.2, 0.4 + 10 ** -10]),
        np.array([0.7, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.2, 0.4]),
    )

    assert (
        opt_mo.get_repeat_length_in_history(
            history_with_three_element_cycle_and_floating_point_error
        )
        == 3
    )


def test_get_repeat_length_in_history_no_cycle():
    history_with_no_cycle = (
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.7, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
        np.array([0.3, 0.4, 0.2, 0.4]),
        np.array([0.7, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3, 0.4]),
    )

    assert opt_mo.get_repeat_length_in_history(history_with_no_cycle) == float(
        "inf"
    )


def test_get_repeat_length_in_history_from_the_beginning():
    history_with_cycle_from_the_beginning = (
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
        np.array([0.3, 0.4, 0.3, 0.3]),
        np.array([0.3, 0.7, 0.3, 0.8]),
    )

    assert (
        opt_mo.get_repeat_length_in_history(
            history_with_cycle_from_the_beginning
        )
        == 2
    )


def test_get_evolutionary_best_response():
    axl.seed(2)
    random_opponents = [[random.random() for _ in range(4)] for _ in range(1)]

    best_ev_response, hist, history_length = opt_mo.get_evolutionary_best_response(
        random_opponents, opt_mo.get_memory_one_best_response
    )

    assert type(best_ev_response) is np.ndarray
    assert len(best_ev_response) == 4
