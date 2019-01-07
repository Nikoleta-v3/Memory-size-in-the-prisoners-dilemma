import numpy as np
import sympy as sym


def mem_one_match_markov_chain(player, opponent):
    """
    Returns a Markov transition matrix for a game of memory one strategies.
    """
    return np.array(
        [
            [
                player[0] * opponent[0],
                player[0] * (1 - opponent[0]),
                (1 - player[0]) * opponent[0],
                (1 - player[0]) * (1 - opponent[0]),
            ],
            [
                player[1] * opponent[2],
                player[1] * (1 - opponent[2]),
                (1 - player[1]) * opponent[2],
                (1 - player[1]) * (1 - opponent[2]),
            ],
            [
                player[2] * opponent[1],
                player[2] * (1 - opponent[1]),
                (1 - player[2]) * opponent[1],
                (1 - player[2]) * (1 - opponent[1]),
            ],
            [
                player[3] * opponent[3],
                player[3] * (1 - opponent[3]),
                (1 - player[3]) * opponent[3],
                (1 - player[3]) * (1 - opponent[3]),
            ],
        ]
    )


def steady_states(matrix, pi):
    """
    A function which returns the stable states of a markov matrix.
    """
    solution = sym.solve(
        [a - b for a, b in zip(matrix.transpose().dot(pi), pi)] + [sum(pi) - 1],
        pi,
    )
    return solution


def make_B(scores, player, opponent):
    """
    A function for creating the B matrix described in the
    literature: 'Iterated Prisoner's Dilemma contains strategies that
    dominate any evolutionary opponent'.
    """
    B = np.array(
        [
            [
                -1 + player[0] * opponent[0],
                -1 + player[0],
                -1 + opponent[0],
                scores[0],
            ],
            [player[1] * opponent[2], -1 + player[1], opponent[2], scores[1]],
            [player[2] * opponent[1], player[2], -1 + opponent[1], scores[2]],
            [player[3] * opponent[3], player[3], opponent[3], scores[3]],
        ]
    )
    return B
