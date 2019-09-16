import numpy as np
import pandas as pd
import sqlalchemy as sa

import opt_mo

df = pd.read_csv("data/numerical_experiments_without_gambler.csv")


gambler = pd.read_csv("data/numerical_experiments_without_gambler.csv")

columns = len(df.columns)
for index in gambler["exp_index"]:
    assert np.allclose(
        gambler[gambler["exp_index"] == index].values[0][:columns],
        df[df["exp_index"] == index].values,
    )
