import sympy as sym
import numpy as np


class MemoryOneStrategy():

    def __init__(self, cc, cd, dc, dd):
        self.cc = cc
        self.cd = cd
        self.dc = dc
        self.dd = dd

    def markov(self, other):
        """
        A method to create the Markov transition matrix for a game of the
        Prisoner's Dilemma.
        """
        return sym.Matrix([
            [self.cc * other.cc, self.cc * (1 - other.cc),
             (1 - self.cc) * other.cc, (1 - self.cc) * (1 - other.cc)],
            [self.cd * other.dc, self.cd * (1 - other.dc),
             (1 - self.cd) * other.dc, (1 - self.cd) * (1 - other.dc)],
            [self.dc * other.cd, self.dc * (1 - other.cd),
             (1 - self.dc) * other.cd, (1 - self.dc) * (1 - other.cd)],
            [self.dd * other.dd, self.dd * (1 - other.dd),
             (1 - self.dd) * other.dd, (1 - self.dd) * (1 - other.dd)]])

def stable_states(matrix, pi):
    """
    A function which returns the stable states of a markov matrix.
    """
    solution = sym.solve([a - b for a, b in zip(matrix.transpose().dot(pi), pi)]
                         + [sum(pi) - 1], pi)
    return solution

def make_B(S, p, q):
    """
    A function for creating the B matrix described in the
    literature: 'Iterated Prisoner's Dilemma contains strategies that
    dominate any evolutionary opponent'.
    """
    B = sym.Matrix([
        [-1 + p[0] * q[0], -1 + p[0], -1 + q[0], S[0]],
        [p[1] * q[2], -1 + p[1], q[2], S[1]],
        [p[2] * q[1], p[2], -1 + q[1], S[2]],
        [p[3] * q[3], p[3], q[3], S[3]],
    ])
    return B