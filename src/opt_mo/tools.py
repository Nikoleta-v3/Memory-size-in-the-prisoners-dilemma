import axelrod as axl
import numpy as np
import pandas as pd
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


def is_ZD(vector, game=axl.game.Game()):
    """
    Check is a strategy p is ZD.
    """
    R, P, S, T = game.RPST()
    tilde_vector = np.array(
        [vector[0] - 1, vector[1] - 1, vector[2], vector[3]]
    )

    expected_tilde_vector1 = (
        P * tilde_vector[1]
        + P * tilde_vector[2]
        - R * tilde_vector[1]
        - R * tilde_vector[2]
    ) / (2 * P - S - T)
    chi = (
        P * tilde_vector[1]
        - P * tilde_vector[2]
        + S * tilde_vector[2]
        - T * tilde_vector[1]
    ) / (
        P * tilde_vector[1]
        - P * tilde_vector[2]
        - S * tilde_vector[1]
        + T * tilde_vector[2]
    )

    return (
        np.isclose(expected_tilde_vector1, tilde_vector[0])
        and chi > 1
        and vector[3] == 0
    )


def get_least_squares(vector, game=axl.game.Game()):
    """
    Obtain the least squares directly
    
    Returns:
    
    - xstar
    - residual
    """

    R, P, S, T = game.RPST()

    C = np.array([[R - P, R - P], [S - P, T - P], [T - P, S - P], [0, 0]])

    tilde_p = np.array([vector[0] - 1, vector[1] - 1, vector[2], vector[3]])

    xstar = np.linalg.inv(C.transpose() @ C) @ C.transpose() @ tilde_p

    SSError = tilde_p.transpose() @ tilde_p - tilde_p @ C @ xstar

    return SSError
