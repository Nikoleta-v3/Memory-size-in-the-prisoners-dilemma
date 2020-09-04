import pathlib

import pandas as pd
import matplotlib.pyplot as plt

import opt_mo


def get_root_of_repo_path():
    """
    Use pathlib to find the root of the repository.
    """
    return pathlib.Path(__file__).absolute().parent.parent.parent


def get_SEE_from_pandas_row(row):
    """
    A vectorized SSE calculation.
    """
    vector = row[
        [
            "Best_response_p1", 
            "Best_response_p2", 
            "Best_response_p3", 
            "Best_response_p4",
        ]
    ]
    return opt_mo.tools.get_least_squares(vector)

def plot_figure(df):
    """
    Plot the figure for a given dataframe of data.
    
    This plots a 1 by 3 figure:
    
    First row:
        - first figure: Violon plot of x_K (4 violons: 1 for each value of K and 1 overall)
        - second figure: Violon plot of SSE (4 violons: 1 for each value of K and 1 overall)
        - third figure: Violon plot of the x_K / non_dynamic_x_K (4 "violons" 1 for each p*_i)
    """

    K_range = sorted(df["K"].unique())
    x_ticks = range(1, 5)
    x_tick_labels = ["1", "2", "3", "All"]
    fig, axarray = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    
    for ax, variable in zip(axarray, ["x_K", "SSE", "ratio"]):
        data = [
            list(df[df["K"] == K][variable]) for K in K_range
        ] + [list(df[variable])]
        ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

        boxes = ax.boxplot(data, showfliers=False)
        for bp in boxes['medians']:
            bp.set_color('red')
            bp.set_linewidth(2)
        for bp in boxes['caps']:
            bp.set_color('black')
            bp.set_linewidth(2)
        for bp in boxes['whiskers']:
            bp.set_color('black')
            bp.set_linewidth(2)
        for bp in boxes['boxes']:
            bp.set_color('black')
            bp.set_linewidth(2)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        if variable == "x_K":
            variable = "$x_K$"
        ax.set_title(variable)
        ax.set_xlabel("$K$")
    
    return fig


root = get_root_of_repo_path()
df = pd.read_csv(root / "moran_data/moran/data.csv")
df["SSE"] = df.apply(get_SEE_from_pandas_row, axis=1)
df["ratio"] = df["x_K"] / df["non_dynamic_x_K"]
df = df.drop_duplicates()
df = df.dropna()

df_with_improved_performance = df[df["x_K"] > df["non_dynamic_x_K"]]
fig = plot_figure(df=df_with_improved_performance);
fig.savefig("main.pdf", transparent=True)
