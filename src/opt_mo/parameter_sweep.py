"""
This script is used to perform numerical experiments. More specifically, for a
given set of opponents calculates the following:

- the best memory one response
- the best evolutionary memory one response
"""

import random
import sys
from collections import namedtuple

import axelrod as axl
import dask
import sqlalchemy as sa

import opt_mo

if __name__ == "__main__":
    # parameters
    seed = int(sys.argv[1])
    num_process = 4
    max_index = seed * 100
    index = (seed - 1) * 100

    engine = sa.create_engine("sqlite:///../data/main.db")
    connection = engine.connect()

    try:
        sql = """
        CREATE TABLE experiments (
            exp_index NUMERIC(5, 0),
            first_opponent_q_1  NUMERIC(6,5),
            first_opponent_q_2  NUMERIC(6,5),
            first_opponent_q_3  NUMERIC(6,5),
            first_opponent_q_4  NUMERIC(6,5),
            second_opponent_q_1  NUMERIC(6,5),
            second_opponent_q_2  NUMERIC(6,5),
            second_opponent_q_3  NUMERIC(6,5),
            second_opponent_q_4  NUMERIC(6,5),
            mem_one_p_1  NUMERIC(6,5),
            mem_one_p_2  NUMERIC(6,5),
            mem_one_p_3  NUMERIC(6,5),
            mem_one_p_4  NUMERIC(6,5),
            evol_mem_one_p_1  NUMERIC(6,5),
            evol_mem_one_p_2  NUMERIC(6,5),
            evol_mem_one_p_3  NUMERIC(6,5),
            evol_mem_one_p_4  NUMERIC(6,5),

            CONSTRAINT experiment_pk PRIMARY KEY (exp_index)
        )
        """
        connection.execute(sql)
    except Exception as e:
        pass

    while index < max_index:
        axl.seed(index)
        opponents = opponents = [
            [random.random() for _ in range(4)] for _ in range(1)
        ]

        jobs = [
            dask.delayed(opt_mo.get_memory_one_best_response)(opponents),
            dask.delayed(opt_mo.get_evolutionary_best_response)(
                opponents, opt_mo.get_memory_one_best_response
            ),
        ]

        results = dask.compute(*jobs, num_workers=num_process)

        values = [
            index,
            opponents[0][0],
            opponents[0][1],
            opponents[0][2],
            opponents[0][3],
            opponents[-1][0],
            opponents[-1][1],
            opponents[-1][2],
            opponents[-1][2],
            results[0][0],
            results[0][1],
            results[0][2],
            results[0][3],
            results[1][0],
            results[1][1],
            results[1][2],
            results[1][3],
        ]

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
                evol_mem_one_p_4)
            VALUES
            """
        sql += "(" + ",".join(["?" for _, in enumerate(values)]) + ")"

        connection.execute(sql, values)

        index += 1
