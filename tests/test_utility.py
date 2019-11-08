import numpy as np

import opt_mo

qs = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0)]


def test_mem_quadratic_numerator():
    assert np.array_equal(
        opt_mo.utility.quadratic_term_numerator(opponent=qs[0]),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )

    assert np.array_equal(
        opt_mo.utility.quadratic_term_numerator(opponent=qs[1]),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )

    assert np.array_equal(
        opt_mo.utility.quadratic_term_numerator(opponent=qs[2]),
        np.array([[0, 0, 1, -5], [0, 0, 0, 3], [1, 0, 0, 0], [-5, 3, 0, 0]]),
    )


def test_mem_linear_numerator():
    assert np.array_equal(
        opt_mo.utility.linear_term_numerator(opponent=qs[0]),
        np.array([0, -1, 0, 0]),
    )
    assert np.array_equal(
        opt_mo.utility.linear_term_numerator(opponent=qs[1]),
        np.array([-5, 0, 3, 0]),
    )
    assert np.array_equal(
        opt_mo.utility.linear_term_numerator(opponent=qs[2]),
        np.array([-1, 0, -1, 5]),
    )


def test_mem_constant_numerator():
    assert opt_mo.utility.constant_term_numerator(opponent=qs[0]) == 1
    assert opt_mo.utility.constant_term_numerator(opponent=qs[1]) == 5
    assert opt_mo.utility.constant_term_numerator(opponent=qs[2]) == 1


qs = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 1, 0)]


def test_mem_quadratic_denominator():
    assert np.array_equal(
        opt_mo.utility.quadratic_term_denominator(opponent=qs[0]),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )
    assert np.array_equal(
        opt_mo.utility.quadratic_term_denominator(opponent=qs[1]),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )
    assert np.array_equal(
        opt_mo.utility.quadratic_term_denominator(opponent=qs[2]),
        np.array([[0, 0, 1, -2], [0, 0, 0, 1], [1, 0, 0, 0], [-2, 1, 0, 0]]),
    )


def test_mem_linear_denominator():
    assert np.array_equal(
        opt_mo.utility.linear_term_denominator(opponent=qs[0]),
        np.array([0, -1, 0, 1]),
    )
    assert np.array_equal(
        opt_mo.utility.linear_term_denominator(opponent=qs[1]),
        np.array([-1, 0, 1, 0]),
    )
    assert np.array_equal(
        opt_mo.utility.linear_term_denominator(opponent=qs[2]),
        np.array([-1, 0, -1, 2]),
    )


def test_mem_constant_denominator():
    assert opt_mo.utility.constant_term_denominator(opponent=qs[0]) == 1
    assert opt_mo.utility.constant_term_denominator(opponent=qs[1]) == 1
    assert opt_mo.utility.constant_term_denominator(opponent=qs[2]) == 1


def test_match_utility_against_defector():
    q = (0, 0, 0, 0)
    assert opt_mo.match_utility(player=(0, 0, 0, 0), opponent=q) == 1
    assert opt_mo.match_utility(player=(1, 1, 1, 1), opponent=q) == 0


def test_match_utility_against_cooperator():
    q = (1, 1, 1, 1)
    assert opt_mo.match_utility(player=(0, 0, 0, 0), opponent=q) == 5
    assert opt_mo.match_utility(player=(1, 1, 1, 1), opponent=q) == 3


def test_match_utility_against_tit_for_tat():
    q = (1, 0, 1, 0)
    assert opt_mo.match_utility(player=(0, 0, 0, 0), opponent=q) == 1
    assert opt_mo.match_utility(player=(1, 1, 1, 1), opponent=q) == 3


def test_tournament_utility_against_defectors():
    q = (0, 0, 0, 0)
    z = (0, 0, 0, 0)
    assert opt_mo.tournament_utility((0, 0, 0, 0), [q, z]) == 1
    assert opt_mo.tournament_utility((1, 1, 1, 1), [q, z]) == 0


def test_tournament_utility_against_cooperators():
    q = (1, 1, 1, 1)
    z = (1, 1, 1, 1)
    assert opt_mo.tournament_utility((0, 0, 0, 0), [q, z]) == 5
    assert opt_mo.tournament_utility((1, 1, 1, 1), [q, z]) == 3


def test_tournament_utility_against_cooperator_defector():
    q = (1, 1, 1, 1)
    z = (0, 0, 0, 0)
    assert opt_mo.tournament_utility((0, 0, 0, 0), [q, z]) == 3
    assert opt_mo.tournament_utility((1, 1, 1, 1), [q, z]) == 1.5


def test_simulate_match_utility_with_defector():
    assert np.isclose(
        opt_mo.simulate_match_utility((0, 0, 0, 0), (0, 0, 0, 0)),
        1,
        atol=10 ** -2,
    )
    assert np.isclose(
        opt_mo.simulate_match_utility((0, 0, 0, 0), (1, 1, 1, 1)),
        5,
        atol=10 ** -2,
    )
    assert np.isclose(
        opt_mo.simulate_match_utility((0, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)),
        3,
        atol=10 ** -2,
    )


def test_simulate_match_utility_with_cooperator():
    assert np.isclose(
        opt_mo.simulate_match_utility((1, 1, 1, 1), (0, 0, 0, 0)),
        0,
        atol=10 ** -2,
    )
    assert np.isclose(
        opt_mo.simulate_match_utility((1, 1, 1, 1), (1, 1, 1, 1)),
        3,
        atol=10 ** -2,
    )
    assert np.isclose(
        opt_mo.simulate_match_utility((1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5)),
        1.5,
        atol=10 ** -2,
    )


def test_simulate_tournament_utility():
    opponents = [(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)]
    player = (1, 0, 1, 0)

    score = opt_mo.simulate_tournament_utility(
        player, opponents, turns=200, repetitions=5
    )

    assert np.isclose(score, 2.5, atol=10 ** -2)
