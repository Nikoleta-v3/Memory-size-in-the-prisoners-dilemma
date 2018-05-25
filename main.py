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
import scipy
import skopt

import axelrod as axl
import opt_mo
from axelrod.action import Action
from axelrod.strategies.lookerup import Plays

import warnings
warnings.filterwarnings("ignore")

C, D = Action.C, Action.D

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

def pattern_size(params):
    """
    Calculates the size of the lookup table.
    """
    depth = [itertools.product((C, D), repeat=i) for i in params]
    keys = itertools.product(depth[0], depth[1], depth[2])

    return len(list(keys))

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

def get_filename(location, params, index, method):
    filename = location + 'gambler{}_{}_{}/'.format(params[0], params[1], params[2])
    filename += '{}_{}.csv'.format(method, index)
    return filename

def get_columns(params, method_params):
    cols = ['index', 'turns', 'repetitions', '$q_1$', '$q_2$', '$q_3$', '$q_4$',
            r'$\bar{q}_1$', r'$\bar{q}_2$', r'$\bar{q}_3$', r'$\bar{q}_4$', '$p_1$',
            '$p_2$', '$p_3$', '$p_4$', '$u_q$', '$U_q$', 'Optimisation time', '$U_{G}$',
            'Training time', 'Gambler Initial']
    size = pattern_size(params)
    gambler_cols = ['Gambler {} key'.format(i) for i in range(size)]
    method_cols = ['method'] + ['{}'.format(key) for key in method_params.keys()]

    return cols + gambler_cols + method_cols

def write_results(index, method, list_opponents, params, turns, repetitions, method_params):

    cols = get_columns(params, method_params)
    frame = pd.DataFrame()

    row = [index]
    row += [turns, repetitions]
    row += [q for player in list_opponents for q in player]
    if len(row) == 7:
        for _ in range(4):
            row.append(None)

    start_optimisation = time.clock()
    best_response, theoretical, simulated = optimal_memory_one(method, opponents=list_opponents,
                                                               turns=turns,repetitions=repetitions,
                                                               method_params=method_params)
    row += [p for p in best_response]
    row.append(theoretical), row.append(simulated)
    row.append(time.clock() - start_optimisation)

    start_training = time.clock()
    opt_gambler, utility = train_gambler(method, opponents=list_opponents, turns=turns,
                                         repetitions=repetitions, params=params,
                                         method_params=method_params)

    row.append(utility), row.append(time.clock() - start_training)
    for vector in opt_gambler:
        row.append(vector)

    row.append(method)
    for value in method_params.values():
        row.append(value)
    frame = frame.append([row])
    frame.columns = cols

    return frame

if __name__ == '__main__':
    num_turns = 200

    index = int(sys.argv[1])
    num_plays = int(sys.argv[2])
    num_op_plays = int(sys.argv[3])
    num_op_start_plays = int(sys.argv[4])
    method = sys.argv[5]

    location = 'data/random_numerical_experiments/'
    params = [num_plays, num_op_plays, num_op_start_plays]

    i = (index - 1) * 100
    while i <= index * 100:
        axl.seed(i)
        main_op = [np.random.random(4)]
        dfs = []
        for num_repetitions in [5, 20, 50]:
 
            filename =  get_filename(location=location, params=params, index=i, method=method)

            if method == 'bayesian':
                for starts, calls in [(10, 20), (20, 30), (20, 40), (20, 45), (20, 50)]:
                    method_params = {'n_random_starts' : starts, 'n_calls': calls}

                    # match
                    dfs.append(write_results(index=i, method=method, list_opponents=main_op,
                                            params=params, turns=num_turns, repetitions=num_repetitions,
                                            method_params=method_params))

                    # tournament
                    axl.seed(i + 10000)
                    other = [np.random.random(4)]

                    opponents = main_op + other
                    dfs.append(write_results(index=i, method=method, list_opponents=opponents,
                                            params=params, turns=num_turns,
                                            repetitions=num_repetitions, method_params=method_params))

            if method == 'differential':
                for popsize, mutation_rate in [(5, 0.1), (10, .2), (15, .2), (20, .3), (25, 0.4)]:
                    method_params = {'popsize' : popsize, 'mutation': mutation_rate}

                    # match
                    dfs.append(write_results(index=i, method=method, list_opponents=main_op,
                                            params=params, turns=num_turns, repetitions=num_repetitions,
                                            method_params=method_params))

                    # tournament
                    axl.seed(i + 10000)
                    other = [np.random.random(4)]

                    opponents = main_op + other
                    dfs.append(write_results(index=i, method=method, list_opponents=opponents,
                                            params=params, turns=num_turns,
                                            repetitions=num_repetitions, method_params=method_params))

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(filename)
        i += 1
