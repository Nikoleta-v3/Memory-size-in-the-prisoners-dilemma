import unittest

import numpy as np
import sympy as sym

import opt_mo

p = (0.1, 0.5, 0.6, 0.7)

q_1, q_2, q_3, q_4 = sym.symbols("q_1, q_2, q_3, q_4")
q = (q_1, q_2, q_3, q_4)


def test_markov_chain():
    M = opt_mo.mem_one_match_markov_chain(p, q)

    assert type(M) is np.ndarray
    assert M.item(0) == 0.1 * q_1
    assert M.item(0) == 0.1 * q_1
    assert M.item(1) == -0.1 * q_1 + 0.1
    assert M.item(2) == 0.9 * q_1
    assert M.item(3) == -0.9 * q_1 + 0.9
    assert M.item(4) == 0.5 * q_3
    assert M.item(5) == -0.5 * q_3 + 0.5
    assert M.item(6) == 0.5 * q_3
    assert M.item(7) == -0.5 * q_3 + 0.5
    assert M.item(8) == 0.6 * q_2
    assert M.item(9) == -0.6 * q_2 + 0.6
    assert M.item(10) == 0.4 * q_2
    assert M.item(11) == -0.4 * q_2 + 0.4
    assert M.item(12) == 0.7 * q_4
    assert M.item(13) == -0.7 * q_4 + 0.7
    assert M.item(14) == (1 - 0.7) * q_4
    assert M.item(15) == (1 - 0.7) * (1 - q_4)


def test_steady_states():
    pi_1, pi_2, pi_3 = sym.symbols("pi_1, pi_2, pi_3")
    pi = (pi_1, pi_2, pi_3)

    matrix = np.array(
        [[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]]
    )
    v = opt_mo.steady_states(matrix, pi)

    assert v[pi_1] == 0.6250
    assert v[pi_2] == 0.3125
    assert v[pi_3] == 0.0625


def test_make_B():
    q_1, q_2, q_3, q_4 = sym.symbols("q_1, q_2, q_3, q_4")
    p_1, p_2, p_3, p_4 = sym.symbols("q_1, q_2, q_3, q_4")

    p = (p_1, p_2, p_3, p_4)
    q = (q_1, q_2, q_3, q_4)
    S = (3, 0, 5, 1)

    B = opt_mo.make_B(S, p, q)

    assert type(B) is np.ndarray
    assert B.item(0) == q_1 ** 2 - 1
    assert B.item(1) == q_1 - 1
    assert B.item(2) == q_1 - 1
    assert B.item(3) == 3
    assert B.item(4) == q_2 * q_3
    assert B.item(5) == q_2 - 1
    assert B.item(6) == q_3
    assert B.item(7) == 0
    assert B.item(8) == q_2 * q_3
    assert B.item(9) == q_3
    assert B.item(10) == q_2 - 1
    assert B.item(11) == 5
    assert B.item(12) == q_4 ** 2
    assert B.item(13) == q_4
    assert B.item(14) == q_4
    assert B.item(15) == 1


def test_is_ZD():
    tit_for_tat = (1, 0, 1, 0)
    assert opt_mo.tools.is_ZD(tit_for_tat) == False

    extort_two = (8 / 9, 1 / 2, 1 / 3, 0)
    assert opt_mo.tools.is_ZD(extort_two) is True


def test_get_least_squares():
    vector = (1, 0, 0, 1)
    assert np.isclose(opt_mo.tools.get_least_squares(vector), 1.23, 10 ** -2)

    extort_two = (8 / 9, 1 / 2, 1 / 3, 0)
    assert np.isclose(opt_mo.tools.get_least_squares(extort_two), 0)
