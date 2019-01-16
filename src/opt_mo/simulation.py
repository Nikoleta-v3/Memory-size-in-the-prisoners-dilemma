import axelrod as axl


def simulate_match(player, opponent, turns=500, repetitions=500):
    total = 0
    players = [axl.MemoryOnePlayer(i) for i in [player, opponent]]
    for rep in range(repetitions):
        match = axl.Match(players=players, turns=turns)
        _ = match.play()

        total += match.final_score_per_turn()[0]

    return total / repetitions
