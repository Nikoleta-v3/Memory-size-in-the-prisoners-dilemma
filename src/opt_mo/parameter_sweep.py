"""
This script is used to perform numerical experiments. More specifically, for a
given set of opponents calculates the following:

- the best memory one response
- the best evolutionary memory one response
- the best reactive response
- the best evolutionary reactive response
"""

import random
import re
import sys
from collections import namedtuple

import axelrod as axl
import dask
import sqlalchemy as sa

import opt_mo

if __name__ == "__main__":

    seed = int(sys.argv[1])
    run_gambler = "run_gambler" in sys.argv

    num_process = 4
    num_of_opponents = 2
    max_index = seed * 100
    start_index = (seed - 1) * 100

    number_of_digits = {
        "digits": 12,
        "floats": 10,
        "index_digits": 10,
        "index_floats": 0,
    }

    if run_gambler:
        folder = "with_gambler"
        params = [1, 1, 2]
        size = opt_mo.get_lookup_table_size(params)
        turns = 500
        repetitions = 200
    else:
        folder = "without_gambler"

    engine = sa.create_engine("sqlite:///data/%s/main.db" % folder)
    connection = engine.connect()

    try:
        sql = """
        CREATE TABLE experiments (
            exp_index NUMERIC({index_digits},{index_floats}),
            first_opponent_q_1  NUMERIC({digits},{floats}),
            first_opponent_q_2  NUMERIC({digits},{floats}),
            first_opponent_q_3  NUMERIC({digits},{floats}),
            first_opponent_q_4  NUMERIC({digits},{floats}),
            second_opponent_q_1  NUMERIC({digits},{floats}),
            second_opponent_q_2  NUMERIC({digits},{floats}),
            second_opponent_q_3  NUMERIC({digits},{floats}),
            second_opponent_q_4  NUMERIC({digits},{floats}),
            mem_one_p_1  NUMERIC({digits},{floats}),
            mem_one_p_2  NUMERIC({digits},{floats}),
            mem_one_p_3  NUMERIC({digits},{floats}),
            mem_one_p_4  NUMERIC({digits},{floats}),
            evol_mem_one_p_1  NUMERIC({digits},{floats}),
            evol_mem_one_p_2  NUMERIC({digits},{floats}),
            evol_mem_one_p_3  NUMERIC({digits},{floats}),
            evol_mem_one_p_4  NUMERIC({digits},{floats}),
            mem_one_cycle_length NUMERIC({index_digits},{index_floats}),
            reactive_p_1  NUMERIC({digits},{floats}),
            reactive_p_2  NUMERIC({digits},{floats}),
            reactive_p_3  NUMERIC({digits},{floats}),
            reactive_p_4  NUMERIC({digits},{floats}),
            evol_reactive_p_1  NUMERIC({digits},{floats}),
            evol_reactive_p_2  NUMERIC({digits},{floats}),
            evol_reactive_p_3  NUMERIC({digits},{floats}),
            evol_reactive_p_4  NUMERIC({digits},{floats}),
            reactive_cycle_length NUMERIC({index_digits},{index_floats}),

            CONSTRAINT experiment_pk PRIMARY KEY (exp_index)
        )
        """.format(
            **number_of_digits
        )

        if run_gambler:
            sql = re.split(r"\n\n", sql)

            gambler_paramater_entries = [
                "\n\tgambler_paramater_p_{}  NUMERIC({digits},{floats}),".format(
                    i, **number_of_digits
                )
                for i in range(size + 1)
            ]

            gambler_paramater_entries.append(
                "\n\tgambler_utility  NUMERIC(20,10),"
            )

            sql = sql[0] + "".join(gambler_paramater_entries) + "\n\n" + sql[1]
        connection.execute(sql)
    except Exception as e:
        pass

    for index in range(start_index, max_index):
        axl.seed(index)
        opponents = [
            [random.random() for _ in range(4)] for _ in range(num_of_opponents)
        ]

        jobs = [
            dask.delayed(opt_mo.get_memory_one_best_response)(opponents),
            dask.delayed(opt_mo.get_evolutionary_best_response)(
                opponents, opt_mo.get_memory_one_best_response
            ),
            dask.delayed(opt_mo.get_reactive_best_response_with_bayesian)(
                opponents
            ),
            dask.delayed(opt_mo.get_evolutionary_best_response)(
                opponents, opt_mo.get_reactive_best_response_with_bayesian
            ),
        ]

        if run_gambler:
            jobs.append(
                dask.delayed(opt_mo.get_best_response_gambler)(
                    opponents,
                    params=params,
                    turns=turns,
                    repetitions=repetitions,
                )
            )
        print("=======START TASKS=======")
        results = dask.compute(*jobs, num_workers=num_process)
        print("=======END TASKS=======")
        values = [
            index,
            opponents[0][0],
            opponents[0][1],
            opponents[0][2],
            opponents[0][3],
            opponents[-1][0],
            opponents[-1][1],
            opponents[-1][2],
            opponents[-1][3],
            results[0][0],
            results[0][1],
            results[0][2],
            results[0][3],
            *results[1][0],
            results[1][-1],
            results[2][0],
            results[2][1],
            results[2][2],
            results[2][3],
            *results[3][0],
            results[3][-1],
        ]
        if run_gambler:
            for v in results[4][0]:
                values.append(v)
            values.append(results[4][-1])

        sql = """
            INSERT INTO experiments
                (exp_index,
                first_opponent_q_1,
                first_opponent_q_2,
                first_opponent_q_3,
                first_opponent_q_4,
                second_opponent_q_1,
                second_opponent_q_2,
                second_opponent_q_3,
                second_opponent_q_4,
                mem_one_p_1,
                mem_one_p_2,
                mem_one_p_3,
                mem_one_p_4,
                evol_mem_one_p_1,
                evol_mem_one_p_2,
                evol_mem_one_p_3,
                evol_mem_one_p_4,
                mem_one_cycle_length,
                reactive_p_1,
                reactive_p_2,
                reactive_p_3,
                reactive_p_4,
                evol_reactive_p_1,
                evol_reactive_p_2,
                evol_reactive_p_3,
                evol_reactive_p_4,
                reactive_cycle_length
            """
        if run_gambler:
            for i in range(size + 1):
                sql += ", \n\tgambler_paramater_p_%s" % i
            sql += ", \n\tgambler_utility"

        sql += (
            ") \n VALUES\n \t("
            + ",".join(["?" for _, i in enumerate(values)])
            + ")"
        )
        values = list(map(float, values))
        try:
            connection.execute(sql, values)
        except ValueError:
            pass
