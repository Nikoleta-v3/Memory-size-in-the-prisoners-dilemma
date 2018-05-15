"""
This script is used to perform numerical experiments. These include,

- approximating optimal memory one strategies
- calculating the theoretical and simulate utility of those strategies
- approximating optimal Gambler type strategy
"""

import itertools
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import scipy.optimize
import skopt

import axelrod as axl
import opt_mo
from axelrod.action import Action
from axelrod.strategies.lookerup import Plays

C, D = Action.C, Action.D

def prepare_objective_bayesian(turns, repetitions, opponents, params):
    objective = partial(objective_score, turns=turns, repetitions=repetitions,
                        params=params, opponents=opponents)
    return objective

def prepare_objective_differential(opponents):
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

def pattern_size(params):
    """
    Calculates the size of the lookup table.
    """
    depth = [itertools.product((C, D), repeat=i) for i in params]
    keys = itertools.product(depth[0], depth[1], depth[2])

    return len(list(keys))

def optimal_memory_one(opponents, turns, repetitions, popsize=700, strategy='best1bin'):
    """
    Approximates the best memory one strategy for the given environment.
    Returns the strategy, the utility u and U.
    """
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    seed = 0
    objective = prepare_objective_differential(opponents=opponents)

    result = scipy.optimize.differential_evolution(func=objective, bounds=bounds,
                                                   strategy=strategy, popsize=popsize,
                                                   seed=seed)
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

def train_gambler(opponents, turns, repetitions, params, n_calls=50, n_random_starts=20):
    """
    Approximates the best gambler for the given environment.
    Returns the strategy and it's utility.
    """
    size = pattern_size(params)
    objective = prepare_objective_bayesian(turns=turns, repetitions=repetitions,
                                           opponents=opponents, params=params)

    res = skopt.gp_minimize(objective, [(0.0, 1.0) for _ in range(size + 1)],
                            acq_func="EI",                    # the acquisition function
                            n_calls=n_calls,                  # the number of evaluations of f
                            n_random_starts=n_random_starts,  # the number of random initialization points
                            random_state=1)                   # the random seed

    return res.x, -res.fun

def get_filename(location, folder, params, index):
    filename = location + 'gambler{}_{}_{}/'.format(params[0], params[1], params[2])
    filename += folder + '/{}.csv'.format(index)
    return filename

def write_results(list_opponents, index, folder, location, params, turns, repetitions):

    cols = ['$q_1$', '$q_2$', '$q_3$', '$q_4$', r'$\bar{q}_1$', r'$\bar{q}_2$',
            r'$\bar{q}_3$', r'$\bar{q}_4$', '$p_1$', '$p_2$', '$p_3$', '$p_4$',
            '$u_q$', '$U_q$', 'Differential evol. time', 'gambler', '$U_{G}$',
            'Gambler train time']
    frame = pd.DataFrame()

    row = [q for player in list_opponents for q in player]
    if len(row) == 4:
        for _ in range(4):
            row.append(None)

    start_differential_evolution = time.clock()
    best_response, theoretical, simulated = optimal_memory_one(opponents=list_opponents,
                                                               turns=turns, repetitions=repetitions)
    row += [p for p in best_response]
    row.append(theoretical), row.append(simulated)
    row.append(time.clock() - start_differential_evolution)

    start_gambler_train = time.clock()
    opt_gambler, utility = train_gambler(opponents=list_opponents, turns=turns,
                                           repetitions=repetitions, params=params)
    row.append(opt_gambler), row.append(utility)
    row.append(time.clock() - start_gambler_train)

    frame = frame.append([row])
    frame.columns = cols

    filename =  get_filename(location, folder, params, index)
    frame.to_csv(filename, index=False)

if __name__ == '__main__':
    num_turns = 200
    num_repetitions = 5
    location = 'data/random_numerical_experiments/'

    index = int(sys.argv[1])
    num_plays = int(sys.argv[2])
    num_op_plays = int(sys.argv[3])
    num_op_start_plays = int(sys.argv[4])

    params = [num_plays, num_op_plays, num_op_start_plays]

    i = (index - 1) * 100
    while i <= index * 100:
        axl.seed(i)
        main_op = [np.random.random(4)]

        # match
        write_results(list_opponents=main_op, folder='matches', index=i,
                      location=location, params=params, turns=num_turns,
                      repetitions=num_repetitions)

        # tournament
        axl.seed(index + 10000)
        op = main_op + [np.random.random(4)]
        write_results(list_opponents=main_op, folder='tournaments', index=i,
                      location=location, params=params, turns=num_turns,
                      repetitions=num_repetitions)

        i += 1
