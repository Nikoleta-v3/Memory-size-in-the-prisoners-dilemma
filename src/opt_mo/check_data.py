import numpy as np

import opt_mo


df = opt_mo.tools.read_sql_data_frame("data/without_gambler/main.db")
df_with_g = opt_mo.tools.read_sql_data_frame("data/with_gambler/main.db")

columns = len(df.columns)
for index in df_with_g["exp_index"]:
    assert np.allclose(
        df_with_g[df_with_g["exp_index"] == index].values[0][:columns],
        df[df["exp_index"] == index].values,
    )
