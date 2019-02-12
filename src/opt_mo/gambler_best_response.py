import itertools
from functools import partial

import axelrod as axl
import numpy as np
import skopt
from axelrod.action import Action
from axelrod.strategies.lookerup import Plays

import opt_mo

C, D = Action.C, Action.D


def get_lookup_table_size(params):
    """
    Calculates the size of the lookup table.
    """
    depth = [itertools.product((C, D), repeat=i) for i in params]
    keys = itertools.product(depth[0], depth[1], depth[2])

    return len(list(keys))


def prepare_objective_training(turns, repetitions, opponents, params):
    objective = partial(
        tournament_score_gambler,
        turns=turns,
        repetitions=repetitions,
        params=params,
        opponents=opponents,
    )
    return objective


def tournament_score_gambler(pattern, turns, repetitions, opponents, params):
    """Calculates the score of a gambler in a tournament."""
    parameters = Plays(
        self_plays=params[0], op_plays=params[1], op_openings=params[2]
    )
    size = get_lookup_table_size(params)

    initial_action = [
        np.random.choice([C, D], p=[pattern[0], 1 - pattern[0]])
        for _ in range(size)
    ]

    player = axl.Gambler(
        pattern=pattern[1:],
        parameters=parameters,
        initial_actions=initial_action,
    )
    opponents = [axl.MemoryOnePlayer(q) for q in opponents]
    players = opponents + [player]

    number_of_players = len(players)
    edges = [(i, number_of_players - 1) for i in range(number_of_players - 1)]
    tournament = axl.Tournament(
        players=players, turns=turns, edges=edges, repetitions=repetitions
    )
    results = tournament.play(progress_bar=False)
    return -np.mean(results.normalised_scores[-1])


def get_best_response_gambler(
    opponents,
    turns,
    repetitions,
    params,
    method_params={"n_random_starts": 20, "n_calls": 40},
    tol=10 ** -5,
    convergence_switch=True,
):
    """
    Approximates the best response Gambler using bayesian optimisation.
    """
    size = get_lookup_table_size(params)
    bounds = [(0, 1.0 - 1 * 10 ** -8) for _ in range(size + 1)]
    objective = prepare_objective_training(
        turns=turns, repetitions=repetitions, opponents=opponents, params=params
    )

    default_calls = method_params["n_calls"]

    while (
        method_params["n_calls"] == default_calls
        or not opt_mo.objective_is_converged(values, tol=tol)
    ) and convergence_switch:
        result = skopt.gp_minimize(
            func=objective,
            dimensions=bounds,
            acq_func="EI",
            random_state=0,
            **method_params
        )

        values = result.func_vals
        method_params["n_calls"] += 20

    return np.array(result.x), -result.fun
