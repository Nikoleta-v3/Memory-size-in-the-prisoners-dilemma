"""
This script is used to perform numerical experiments. These include,

- approximating optimal memory one strategies
- calculating the theoretical and simulate utility of those strategies
- approximating optimal Gambler type strategy
"""

import sys
import time
import warnings

import axelrod as axl
import numpy as np
import pandas as pd

import opt_mo

warnings.filterwarnings("ignore")


def get_filename(location, params, index):
    filename = location + "gambler{}_{}_{}/".format(
        params[0], params[1], params[2]
    )
    filename += "{}.csv".format(index)
    return filename


def get_columns(params, method_params):
    cols = [
        "index",
        "turns",
        "repetitions",
        "$q_1$",
        "$q_2$",
        "$q_3$",
        "$q_4$",
        r"$\bar{q}_1$",
        r"$\bar{q}_2$",
        r"$\bar{q}_3$",
        r"$\bar{q}_4$",
        "$p_1$",
        "$p_2$",
        "$p_3$",
        "$p_4$",
        "$u_q$",
        "$U_q$",
        "Optimisation time",
        "$U_{G}$",
        "Training time",
        "Gambler Initial",
    ]
    size = opt_mo.pattern_size(params)
    gambler_cols = ["Gambler {} key".format(i) for i in range(size)]
    method_cols = ["{}".format(key) for key in method_params.keys()]

    return cols + gambler_cols + method_cols


def write_results(
    index, list_opponents, params, turns, repetitions, method_params
):

    cols = get_columns(params, method_params)
    frame = pd.DataFrame()

    row = [index]
    row += [turns, repetitions]
    row += [q for player in list_opponents for q in player]
    if len(row) == 7:
        for _ in range(4):
            row.append(None)

    start_optimisation = time.clock()
    best_response, theoretical, simulated = opt_mo.optimal_memory_one(
        opponents=list_opponents,
        turns=turns,
        repetitions=repetitions,
        method_params=method_params,
    )
    row += [p for p in best_response]
    row.append(theoretical), row.append(simulated)
    row.append(time.clock() - start_optimisation)

    start_training = time.clock()
    opt_gambler, utility = opt_mo.train_gambler(
        opponents=list_opponents,
        turns=turns,
        repetitions=repetitions,
        params=params,
        method_params=method_params,
    )

    row.append(utility), row.append(time.clock() - start_training)
    for vector in opt_gambler:
        row.append(vector)

    for value in method_params.values():
        row.append(value)
    frame = frame.append([row])
    frame.columns = cols

    return frame


if __name__ == "__main__":
    num_turns = 200

    index = int(sys.argv[1])
    num_plays = int(sys.argv[2])
    num_op_plays = int(sys.argv[3])
    num_op_start_plays = int(sys.argv[4])

    location = "data/random_numerical_experiments/"
    params = [num_plays, num_op_plays, num_op_start_plays]

    i = (index - 1) * 100
    while i <= index * 100:
        axl.seed(i)
        main_op = [np.random.random(4)]
        dfs = []
        for num_repetitions in [5, 20, 50]:

            filename = get_filename(location=location, params=params, index=i)

            for starts, calls in [
                (10, 20),
                (20, 30),
                (20, 40),
                (20, 45),
                (20, 50),
            ]:
                method_params = {"n_random_starts": starts, "n_calls": calls}

                # match
                dfs.append(
                    write_results(
                        index=i,
                        list_opponents=main_op,
                        params=params,
                        turns=num_turns,
                        repetitions=num_repetitions,
                        method_params=method_params,
                    )
                )

                # tournament
                axl.seed(i + 10000)
                other = [np.random.random(4)]

                opponents = main_op + other
                dfs.append(
                    write_results(
                        index=i,
                        list_opponents=opponents,
                        params=params,
                        turns=num_turns,
                        repetitions=num_repetitions,
                        method_params=method_params,
                    )
                )

                # tournament
                axl.seed(i + 10000)
                other = [np.random.random(4)]

                opponents = main_op + other
                dfs.append(
                    write_results(
                        index=i,
                        list_opponents=opponents,
                        params=params,
                        turns=num_turns,
                        repetitions=num_repetitions,
                        method_params=method_params,
                    )
                )

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(filename)
        i += 1
