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


def test_get_evolutionary_best_response_against_random_opponent():
    axl.seed(2)
    random_opponents = [[random.random() for _ in range(4)]]

    best_ev_response, hist, history_length = opt_mo.get_evolutionary_best_response(
        random_opponents, opt_mo.get_memory_one_best_response
    )

    expected_best_response = np.array([0.15993435, 0, 0, 0])
    assert type(best_ev_response) is np.ndarray
    assert np.allclose(expected_best_response, best_ev_response)


def test_get_evolutionary_best_response_with_K_equal_2_against_random_opponent():
    axl.seed(2)
    random_opponents = [[random.random() for _ in range(4)]]

    best_ev_response, hist, history_length = opt_mo.get_evolutionary_best_response(
        random_opponents, opt_mo.get_memory_one_best_response, K=2,
    )

    expected_best_response = np.array([0.11235944, 0, 0, 0])
    assert type(best_ev_response) is np.ndarray
    assert len(best_ev_response) == 4
    assert np.allclose(expected_best_response, best_ev_response)


def test_get_evolutionary_best_response_with_K_equal_4_against_random_opponent():
    axl.seed(2)
    random_opponents = [[random.random() for _ in range(4)]]

    best_ev_response, hist, history_length = opt_mo.get_evolutionary_best_response(
        random_opponents, opt_mo.get_memory_one_best_response, K=4,
    )

    expected_best_response = np.array([0.01547687, 0, 0, 0])
    assert type(best_ev_response) is np.ndarray
    assert len(best_ev_response) == 4
    assert np.allclose(expected_best_response, best_ev_response, atol=1e-3)


def test_get_evolutionary_best_response_with_K_equal_1_against_2_defectors():
    """
    We see that the expected best response against 2 defectors is:

    [1, 0.45420466, 0, 0.03799109]

    - the first term: indicates that when the player is playing against itself
      it knows to always cooperate.
    - The second term (P_CD): it will still cooperate with relatively high
      probability (0.454...).
    - The third term (P_DC): it will never cooperate after defecting against one
      a player like itself.
    - The final term (P_DD): it is very likely to defect on the occasions where
      it defected and so did it's opponent (probably likely that the opponent is
      defector?).
    """
    opponents = [np.array([0, 0, 0, 0])] * 2

    axl.seed(2)
    best_ev_response, hist, history_length = opt_mo.get_evolutionary_best_response(
        opponents, opt_mo.get_memory_one_best_response, K=1,
    )

    expected_best_response = np.array([1, 0.45420466, 0, 0.03799109])
    assert type(best_ev_response) is np.ndarray
    assert len(best_ev_response) == 4
    assert np.allclose(expected_best_response, best_ev_response)


def test_get_evolutionary_best_response_with_K_equal_10_against_2_defectors():
    """
    With K = 10 and 2 defectors the best response changes from `[1, 0.45420466,
    0, 0.03799109]` (for K = 1) to:

    [0.26874769, 0.11697721, 0, 0]

    TODO: Nik does this make sense?
    """
    opponents = [np.array([0, 0, 0, 0])] * 2

    axl.seed(2)
    best_ev_response, hist, history_length = opt_mo.get_evolutionary_best_response(
        opponents, opt_mo.get_memory_one_best_response, K=10,
    )

    expected_best_response = np.array([0.26874769, 0.11697721, 0, 0])
    assert type(best_ev_response) is np.ndarray
    assert len(best_ev_response) == 4
    assert np.allclose(expected_best_response, best_ev_response, atol=1e-3)
