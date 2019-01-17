import axelrod as axl
import numpy as np


def simulate_match(player, opponent, turns=500, repetitions=200):
    total = 0
    players = [axl.MemoryOnePlayer(i) for i in [player, opponent]]
    for rep in range(repetitions):
        match = axl.Match(players=players, turns=turns)
        _ = match.play()

        total += match.final_score_per_turn()[0]

    return total / repetitions


def simulate_spatial_tournament(player, opponents, turns=500, repetitions=200):

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
