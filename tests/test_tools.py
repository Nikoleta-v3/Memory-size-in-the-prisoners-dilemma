import unittest
import opt_mo.tools as tools

import sympy as sym


class TestTransitions(unittest.TestCase):
    p = tools.MemoryOneStrategy(0.1, 0.5, 0.6, 0.7)

    q_1, q_2, q_3, q_4 = sym.symbols("q_1, q_2, q_3, q_4")
    q = tools.MemoryOneStrategy(q_1, q_2, q_3, q_4)

    def test_init(self):
        self.assertEqual(self.p.cc, 0.1)
        self.assertEqual(self.p.cd, 0.5)
        self.assertEqual(self.p.dc, 0.6)
        self.assertEqual(self.p.dd, 0.7)

        self.assertEqual(self.q.cc, self.q_1)
        self.assertEqual(self.q.cd, self.q_2)
        self.assertEqual(self.q.dc, self.q_3)
        self.assertEqual(self.q.dd, self.q_4)

    def test_markov(self):
        M = self.p.markov(self.q)

        self.assertIsInstance(M, sym.Matrix)
        self.assertEqual(M[0], 0.1 * self.q_1)
        self.assertEqual(M[0], 0.1 * self.q_1)
        self.assertEqual(M[1], -0.1 * self.q_1 + 0.1)
        self.assertEqual(M[2], 0.9 * self.q_1)
        self.assertEqual(M[3], -0.9 * self.q_1 + 0.9)
        self.assertEqual(M[4], 0.5 * self.q_3)
        self.assertEqual(M[5], -0.5 * self.q_3 + 0.5)
        self.assertEqual(M[6], 0.5 * self.q_3)
        self.assertEqual(M[7], -0.5 * self.q_3 + 0.5)
        self.assertEqual(M[8], 0.6 * self.q_2)
        self.assertEqual(M[9], -0.6 * self.q_2 + 0.6)
        self.assertEqual(M[10], 0.4 * self.q_2)
        self.assertEqual(M[11], -0.4 * self.q_2 + 0.4)
        self.assertEqual(M[12], 0.7 * self.q_4)
        self.assertEqual(M[13], -0.7 * self.q_4 + 0.7)
        self.assertEqual(M[14], (1 - 0.7) * self.q_4)
        self.assertEqual(M[15], (1 - 0.7) * (1 - self.q_4))


class TestStates(unittest.TestCase):

    def test_stable_states(self):
        pi_1, pi_2, pi_3 = sym.symbols('pi_1, pi_2, pi_3')
        pi = (pi_1, pi_2, pi_3)

        matrix = sym.Matrix([[0.9, 0.075, 0.025],
                             [0.15, 0.8, 0.05],
                             [0.25, 0.25, 0.5]])
        v = tools.stable_states(matrix, pi)

        self.assertEqual(v[pi_1], 0.6250)
        self.assertEqual(v[pi_2], 0.3125)
        self.assertEqual(v[pi_3], 0.0625)


class TestMakeB(unittest.TestCase):

    q_1, q_2, q_3, q_4 = sym.symbols("q_1, q_2, q_3, q_4")
    p_1, p_2, p_3, p_4 = sym.symbols("q_1, q_2, q_3, q_4")

    p = (p_1, p_2, p_3, p_4)
    q = (q_1, q_2, q_3, q_4)
    S = (3, 0, 5, 1)

    B = tools.make_B(S, p, q)

    def test_make_B(self):

        self.assertIsInstance(self.B, sym.Matrix)
        self.assertEqual(self.B[0], self.q_1 ** 2 - 1)
        self.assertEqual(self.B[1], self.q_1 - 1)
        self.assertEqual(self.B[2], self.q_1 - 1)
        self.assertEqual(self.B[3], 3)
        self.assertEqual(self.B[4], self.q_2 * self.q_3)
        self.assertEqual(self.B[5], self.q_2 - 1)
        self.assertEqual(self.B[6], self.q_3)
        self.assertEqual(self.B[7], 0)
        self.assertEqual(self.B[8], self.q_2 * self.q_3)
        self.assertEqual(self.B[9], self.q_3)
        self.assertEqual(self.B[10], self.q_2 - 1)
        self.assertEqual(self.B[11], 5)
        self.assertEqual(self.B[12], self.q_4 ** 2)
        self.assertEqual(self.B[13], self.q_4)
        self.assertEqual(self.B[14], self.q_4)
        self.assertEqual(self.B[15], 1)

