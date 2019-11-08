import axelrod as axl
import numpy as np


def quadratic_term_numerator(opponent):
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


def linear_term_numerator(opponent):
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


def constant_term_numerator(opponent):
    """
    Returns the constant part $a$ for memory one strategies.
    """
    constant = -opponent[1] + 5 * opponent[3] + 1

    return constant


def quadratic_term_denominator(opponent):
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


def linear_term_denominator(opponent):
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


def constant_term_denominator(opponent):
    """
    Returns the constant part $\bar{a}$ for memory one strategies.
    """
    constant = -opponent[1] + opponent[3] + 1

    return constant


def match_utility(player, opponent):
    """
    Returns the utility of a player for a given opponent.
    """
    x = np.array(player)

    Q = quadratic_term_numerator(opponent)
    c = linear_term_numerator(opponent)
    a = constant_term_numerator(opponent)

    numerator = np.dot(x, Q.dot(x.T) * 1 / 2) + np.dot(c, x.T) + a

    Q_bar = quadratic_term_denominator(opponent)
    c_bar = linear_term_denominator(opponent)
    a_bar = constant_term_denominator(opponent)

    denominator = np.dot(x, Q_bar.dot(x.T) * 1 / 2) + np.dot(c_bar, x.T) + a_bar

    return numerator / denominator


def tournament_utility(player, opponents):
    """
    Returns the negative utility of a player against a list of opponents.
    The tournament utility of a player is the sum of utilities against each
    opponent and against itself.
    """
    obj = 0
    return np.mean([match_utility(player, opponent) for opponent in opponents])


def simulate_match_utility(player, opponent, turns=500, repetitions=200):
    """
    Returns the simulated utility of a memory one player against a single opponent.
    """
    total = 0
    players = [axl.MemoryOnePlayer(vector) for vector in [player, opponent]]
    for rep in range(repetitions):
        match = axl.Match(players=players, turns=turns)
        _ = match.play()

        total += match.final_score_per_turn()[0]

    return total / repetitions


def simulate_tournament_utility(player, opponents, turns=500, repetitions=200):
    """
    Returns the simulated utility of a memory one strategy in a tournament.
    """

    strategies = [axl.MemoryOnePlayer(p) for p in opponents] + [
        axl.MemoryOnePlayer(player)
    ]
    number_of_players = len(strategies)
    edges = [(i, number_of_players - 1) for i in range(number_of_players - 1)]
    tournament = axl.Tournament(
        players=strategies, turns=turns, repetitions=repetitions, edges=edges
    )
    results = tournament.play(progress_bar=False)

    return np.mean(results.normalised_scores[-1])
