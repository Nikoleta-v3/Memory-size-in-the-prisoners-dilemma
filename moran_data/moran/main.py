import csv

import axelrod as axl
import numpy as np
import pandas as pd

import opt_mo


def theoretic_fixation(N, payoff_matrix, K=1):
    """
    Calculate x_i assuming a non dynamic player
    """
    f_ones = np.array(
        [
            (payoff_matrix[0, 0] * (K - 1) + payoff_matrix[0, 1] * (N - K))
            for K in range(1, N)
        ]
    )
    f_twos = np.array(
        [
            (payoff_matrix[1, 0] * K + payoff_matrix[1, 1] * (N - K - 1))
            for K in range(1, N)
        ]
    )
    gammas = f_twos / f_ones
    return (1 + np.sum(np.cumprod(gammas[: K - 1]))) / (
        1 + np.sum(np.cumprod(gammas))
    )


def obtain_payoff_matrix(opponent, K, N=4):
    """
    Obtain the payoff matrix for K best response players in a population of a
    total of N individuals.
    """
    best_ev_response, _, _ = opt_mo.get_evolutionary_best_response(
        [opponent] * (N - K), opt_mo.get_memory_one_best_response, K=K
    )
    players = [best_ev_response, opponent]
    return (
        np.array(
            [
                [opt_mo.match_utility(player, opponent) for opponent in players]
                for player in players
            ]
        ),
        best_ev_response,
    )


def best_response_moran_process(opponent, N=4):
    payoff_matrices = {}
    best_response_players = {}
    for K in range(1, N):
        payoff_matrix, best_response = obtain_payoff_matrix(
            opponent=opponent, K=K
        )
        payoff_matrices[K] = payoff_matrix
        best_response_players[K] = best_response

    f_ones = np.array(
        [
            (
                payoff_matrices[K][0, 0] * (K - 1)
                + payoff_matrices[K][0, 1] * (N - K)
            )
            for K in range(1, N)
        ]
    )
    f_twos = np.array(
        [
            (
                payoff_matrices[K][1, 0] * K
                + payoff_matrices[K][1, 1] * (N - K - 1)
            )
            for K in range(1, N)
        ]
    )
    gammas = f_twos / f_ones
    return (
        {
            K: (1 + np.sum(np.cumprod(gammas[: K - 1])))
            / (1 + np.sum(np.cumprod(gammas)))
            for K in range(1, N)
        },
        payoff_matrices,
        best_response_players,
    )


if __name__ == "__main__":
    N = 4
    try:
        df = pd.read_csv("data.csv")
        try:
            seed = int(df["seed"].max())
        except ValueError:
            seed = 0
    except FileNotFoundError:
        header = [
            "seed",
            "Opponent_p1",
            "Opponent_p2",
            "Opponent_p3",
            "Opponent_p4",
            "K",
            "A_11",
            "A_12",
            "A_21",
            "A_22",
            "Best_response_p1",
            "Best_response_p2",
            "Best_response_p3",
            "Best_response_p4",
            "x_K",
            "non_dynamic_x_K",
        ]
        seed = 0
        with open("data.csv", "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)

    while True:
        with open("data.csv", "a") as f:
            csv_writer = csv.writer(f)

            np.random.seed(seed)
            opponent = np.random.random(4)
            (
                fixation_probabilities,
                payoff_matrices,
                best_response_players,
            ) = best_response_moran_process(opponent, N=N)

            row = [seed] + list(opponent)
            for K in range(1, N):
                payoff_matrix = list(payoff_matrices[K].flatten())
                best_response = list(best_response_players[K])
                x_K = fixation_probabilities[K]
                non_dynamic_x_K = theoretic_fixation(
                    payoff_matrix=payoff_matrices[K], N=N, K=K
                )
                csv_writer.writerow(
                    row
                    + [K]
                    + payoff_matrix
                    + best_response
                    + [x_K, non_dynamic_x_K]
                )

            seed += 1
