import numpy as np
import pandas as pd
import sqlalchemy as sa

import opt_mo

engine_df = sa.create_engine("sqlite:///data/without_gambler/main.db")
df = pd.read_sql("experiments", engine_df.connect())


engine_df_with_gambler = sa.create_engine("sqlite:///data/with_gambler/main.db")
df_with_gambler = pd.read_sql("experiments", engine_df_with_gambler.connect())

columns = len(df.columns)
for index in df_with_gambler["exp_index"]:
    assert np.allclose(
        df_with_gambler[df_with_gambler["exp_index"] == index].values[0][
            :columns
        ],
        df[df["exp_index"] == index].values,
    )
