"""
This file contains methods used in optimising but also training strategies.
"""
import itertools
from functools import partial

import axelrod as axl
import numpy as np
import scipy
import skopt
from axelrod.action import Action
from axelrod.strategies.lookerup import Plays

import opt_mo

C, D = Action.C, Action.D

def pattern_size(params):
    """
    Calculates the size of the lookup table.
    """
    depth = [itertools.product((C, D), repeat=i) for i in params]
    keys = itertools.product(depth[0], depth[1], depth[2])

    return len(list(keys))

def prepare_objective_training(turns, repetitions, opponents, params):
    objective = partial(objective_score, turns=turns, repetitions=repetitions,
                        params=params, opponents=opponents)
    return objective

def prepare_objective_optimisation(opponents):
    objective = partial(opt_mo.tournament_utility, opponents=opponents)
    return objective

def objective_score(pattern, turns, repetitions, opponents, params):
    """Objective function to maximize total score over matches."""
    parameters = Plays(self_plays=params[0], op_plays=params[1], op_openings=params[2])
    size = pattern_size(params)

    initial_action = [np.random.choice([C, D], p=[pattern[0], 1 - pattern[0]])
                      for _ in range(size)]

    player = axl.Gambler(pattern=pattern[1:], parameters=parameters,
                         initial_actions=initial_action)
    opponents = [axl.MemoryOnePlayer(q) for q in opponents]
    players = opponents + [player]

    number_of_players = len(players)
    edges = [(i, number_of_players - 1) for i in range(number_of_players - 1)]
    tournament = axl.Tournament(players=players, turns=turns, edges=edges,
                                repetitions=repetitions)
    results = tournament.play(progress_bar=False)
    return - np.mean(results.normalised_scores[-1])

def optimal_memory_one(method, opponents, turns, repetitions, method_params):
    """
    Approximates the best memory one strategy for the given environment.
    Returns the strategy, the utility u and U.
    """
    bounds = [(0, 0.9999) for _ in range(4)]
    objective = prepare_objective_optimisation(opponents=opponents)

    if method == 'bayesian':
        result = skopt.gp_minimize(func=objective, dimensions=bounds,
                                   acq_func="EI", random_state=0, **method_params)
    if method == 'differential':
        print('start')
        result = scipy.optimize.differential_evolution(func=objective, bounds=bounds,
                                                       strategy='best1bin', seed=0,
                                                       **method_params)

    best_response = list(result.x)

    mem_players = [axl.MemoryOnePlayer(i) for i in opponents]
    mem_players.append(axl.MemoryOnePlayer(best_response))

    number_of_players = len(mem_players)
    edges = [(i, number_of_players - 1) for i in range(number_of_players - 1)]

    tournament = axl.Tournament(players=mem_players, turns=turns, edges=edges,
                                repetitions=repetitions)
    results = tournament. play(progress_bar=False)
    score = np.mean(results.normalised_scores[-1])

    return best_response, -result.fun, score

def train_gambler(method, opponents, turns, repetitions, params, method_params):
    """
    Approximates the best gambler for the given environment.
    Returns the strategy and it's utility.
    """
    size = pattern_size(params)
    bounds = [(0.0, .99999) for _ in range(size + 1)]
    objective = prepare_objective_training(turns=turns, repetitions=repetitions,
                                           opponents=opponents, params=params)

    if method == 'bayesian':
        result = skopt.gp_minimize(func=objective, dimensions=bounds,
                                   acq_func="EI", random_state=0, **method_params)
    if method == 'differential':
        print('start')
        result = scipy.optimize.differential_evolution(func=objective, bounds=bounds,
                                                       strategy='best1bin', seed=0,
                                                       **method_params)
    return result.x, -result.fun
