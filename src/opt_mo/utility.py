import numpy as np


def mem_quadratic_numerator(opponent):
    """
    Returns the matrix $Q$ for memory one strategies.
    """
    matrix = np.array(
        [
            [
                0,
                -opponent[0] * opponent[1]
                + 5 * opponent[0] * opponent[3]
                + opponent[0]
                + opponent[1] * opponent[2]
                - 5 * opponent[2] * opponent[3]
                - opponent[2],
                opponent[0] * opponent[2] - opponent[1] * opponent[2],
                -5 * opponent[0] * opponent[2] + 5 * opponent[2] * opponent[3],
            ],
            [
                -opponent[0] * opponent[1]
                + 5 * opponent[0] * opponent[3]
                + opponent[0]
                + opponent[1] * opponent[2]
                - 5 * opponent[2] * opponent[3]
                - opponent[2],
                0,
                opponent[0] * opponent[1]
                - opponent[0] * opponent[2]
                - 3 * opponent[1] * opponent[3]
                - opponent[1]
                + 3 * opponent[2] * opponent[3]
                + opponent[2],
                5 * opponent[0] * opponent[2]
                - 5 * opponent[0] * opponent[3]
                - 3 * opponent[1] * opponent[2]
                + 3 * opponent[1] * opponent[3]
                - 2 * opponent[2]
                + 2 * opponent[3],
            ],
            [
                opponent[0] * opponent[2] - opponent[1] * opponent[2],
                opponent[0] * opponent[1]
                - opponent[0] * opponent[2]
                - 3 * opponent[1] * opponent[3]
                - opponent[1]
                + 3 * opponent[2] * opponent[3]
                + opponent[2],
                0,
                3 * opponent[1] * opponent[2] - 3 * opponent[2] * opponent[3],
            ],
            [
                -5 * opponent[0] * opponent[2] + 5 * opponent[2] * opponent[3],
                5 * opponent[0] * opponent[2]
                - 5 * opponent[0] * opponent[3]
                - 3 * opponent[1] * opponent[2]
                + 3 * opponent[1] * opponent[3]
                - 2 * opponent[2]
                + 2 * opponent[3],
                3 * opponent[1] * opponent[2] - 3 * opponent[2] * opponent[3],
                0,
            ],
        ]
    )
    return matrix


def mem_linear_numerator(opponent):
    """
    Returns the matrix $c$ for memory one strategies.
    """
    matrix = np.array(
        [
            opponent[0] * opponent[1]
            - 5 * opponent[0] * opponent[3]
            - opponent[0],
            -opponent[1] * opponent[2]
            + opponent[1]
            + 5 * opponent[2] * opponent[3]
            + opponent[2]
            - 5 * opponent[3]
            - 1,
            -opponent[0] * opponent[1]
            + opponent[1] * opponent[2]
            + 3 * opponent[1] * opponent[3]
            + opponent[1]
            - opponent[2],
            5 * opponent[0] * opponent[3]
            - 3 * opponent[1] * opponent[3]
            - 5 * opponent[2] * opponent[3]
            + 5 * opponent[2]
            - 2 * opponent[3],
        ]
    )
    return matrix


def mem_constant_numerator(opponent):
    """
    Returns the constant part $a$ for memory one strategies.
    """
    constant = -opponent[1] + 5 * opponent[3] + 1

    return constant


def mem_quadratic_denominator(opponent):
    """
    Returns the matrix $\bar{Q}$ for memory one strategies.
    """
    matrix = np.array(
        [
            [
                0,
                -opponent[0] * opponent[1]
                + opponent[0] * opponent[3]
                + opponent[0]
                + opponent[1] * opponent[2]
                - opponent[2] * opponent[3]
                - opponent[2],
                opponent[0] * opponent[2]
                - opponent[0] * opponent[3]
                - opponent[1] * opponent[2]
                + opponent[1] * opponent[3],
                opponent[0] * opponent[1]
                - opponent[0] * opponent[2]
                - opponent[0]
                - opponent[1] * opponent[3]
                + opponent[2] * opponent[3]
                + opponent[3],
            ],
            [
                -opponent[0] * opponent[1]
                + opponent[0] * opponent[3]
                + opponent[0]
                + opponent[1] * opponent[2]
                - opponent[2] * opponent[3]
                - opponent[2],
                0,
                opponent[0] * opponent[1]
                - opponent[0] * opponent[2]
                - opponent[1] * opponent[3]
                - opponent[1]
                + opponent[2] * opponent[3]
                + opponent[2],
                opponent[0] * opponent[2]
                - opponent[0] * opponent[3]
                - opponent[1] * opponent[2]
                + opponent[1] * opponent[3],
            ],
            [
                opponent[0] * opponent[2]
                - opponent[0] * opponent[3]
                - opponent[1] * opponent[2]
                + opponent[1] * opponent[3],
                opponent[0] * opponent[1]
                - opponent[0] * opponent[2]
                - opponent[1] * opponent[3]
                - opponent[1]
                + opponent[2] * opponent[3]
                + opponent[2],
                0,
                -opponent[0] * opponent[1]
                + opponent[0] * opponent[3]
                + opponent[1] * opponent[2]
                + opponent[1]
                - opponent[2] * opponent[3]
                - opponent[3],
            ],
            [
                opponent[0] * opponent[1]
                - opponent[0] * opponent[2]
                - opponent[0]
                - opponent[1] * opponent[3]
                + opponent[2] * opponent[3]
                + opponent[3],
                opponent[0] * opponent[2]
                - opponent[0] * opponent[3]
                - opponent[1] * opponent[2]
                + opponent[1] * opponent[3],
                -opponent[0] * opponent[1]
                + opponent[0] * opponent[3]
                + opponent[1] * opponent[2]
                + opponent[1]
                - opponent[2] * opponent[3]
                - opponent[3],
                0,
            ],
        ]
    )
    return matrix


def mem_linear_denominator(opponent):
    """
    Returns the matrix $\bar{c}$ for memory one strategies.
    """
    matrix = np.array(
        [
            opponent[0] * opponent[1] - opponent[0] * opponent[3] - opponent[0],
            -opponent[1] * opponent[2]
            + opponent[1]
            + opponent[2] * opponent[3]
            + opponent[2]
            - opponent[3]
            - 1,
            -opponent[0] * opponent[1]
            + opponent[1] * opponent[2]
            + opponent[1]
            - opponent[2]
            + opponent[3],
            opponent[0] * opponent[3]
            - opponent[1]
            - opponent[2] * opponent[3]
            + opponent[2]
            - opponent[3]
            + 1,
        ]
    )
    return matrix


def mem_constant_denominator(opponent):
    """
    Returns the constant part $\bar{a}$ for memory one strategies.
    """
    constant = -opponent[1] + opponent[3] + 1

    return constant


def utility(player, opponent):
    """
    Returns the utility of a player for a given opponent.
    """
    x = np.array(player)

    Q = mem_quadratic_numerator(opponent)
    c = mem_linear_numerator(opponent)
    a = mem_constant_numerator(opponent)

    numerator = np.dot(x, Q.dot(x.T) * 1 / 2) + np.dot(c, x.T) + a

    Q_d = mem_quadratic_denominator(opponent)
    d = mem_linear_denominator(opponent)
    b = mem_constant_denominator(opponent)

    denominator = np.dot(x, Q_d.dot(x.T) * 1 / 2) + np.dot(d, x.T) + b

    return numerator / denominator


def tournament_utility(player, opponents):
    """
    Returns the negative utility of a player against a list of opponents.
    The tournament utility of a player is the sum of utilities against each
    opponent and against itself.
    """
    obj = 0  # utility(player, player)
    for opponent in opponents:
        obj += utility(player, opponent)
    return -obj / (len(opponents))
