import axelrod as axl
import dask
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import opt_mo

font = {"size": 10, "weight": "bold"}
matplotlib.rc("font", **font)

df = opt_mo.tools.read_sql_data_frame("data/without_gambler/main.db")


def plot_Ashlock_fingerprints(
    data, step, points, filename=False, cmap="seismic", interpolation="none"
):
    """
    A function for costum plotting Ashlock fingerprints from the data.
    """
    size = int((1 / step) // 1) + 1
    titles = ["Best response", "Evo"]

    fig, axes = plt.subplots(ncols=2, figsize=(10, 6.5))
    for i, dt in enumerate(data):
        plotting_data = axl.fingerprint._reshape_data(dt, points, size)

        im = axes[i].imshow(
            plotting_data, cmap=cmap, interpolation=interpolation
        )

        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.2)

        max_score = max(dt.values())
        min_score = min(dt.values())
        ticks = [min_score, (max_score + min_score) / 2, max_score]
        fig.colorbar(im, cax=cax, ticks=ticks, ax=axes[i])

        axes[i].set_xlabel("$x$")
        axes[i].set_ylabel("$y$", rotation=0)
        axes[i].tick_params(axis="both", which="both", length=0)
        axes[i].set_title(titles[i])

    plt.setp(
        axes,
        xticks=[0, len(plotting_data) - 1],
        xticklabels=["0", "1"],
        yticks=[0, len(plotting_data) - 1],
        yticklabels=["1", "0"],
    )
    plt.subplots_adjust(right=1.3)

    if filename:
        fig.savefig(filename, bbox_inches="tight")

    return fig


def get_Ashlock_fingerprints(best_response, ev_best_response, index):
    """
    Retrieves and plots the Ashlock_fingerprints for the best response strategy
    and it's equivalent evo.
    """
    data = []
    af = axl.AshlockFingerprint(best_response, probe)
    data.append(
        af.fingerprint(turns=10, repetitions=2, step=0.05, progress_bar=False)
    )

    af = axl.AshlockFingerprint(ev_best_response, probe)
    data.append(
        af.fingerprint(turns=10, repetitions=2, step=0.05, progress_bar=False)
    )

    filename = "img/Fingerprints/AshlockFingerprint_index_%s" % index
    plot_Ashlock_fingerprints(
        data=data, step=step, points=points, filename=filename
    )


def get_transitive_fingerprints(best_response, ev_best_response, index):
    """
    Retrieves and plots the Transitive fingerprints for the best response strategy
    and it's equivalent evo.
    """
    fig, axes = plt.subplots(ncols=2, figsize=(10, 6.5))

    tf = axl.TransitiveFingerprint(best_response)
    data = tf.fingerprint(turns=turns, progress_bar=False)
    tf.plot(ax=axes[0])

    tf = axl.TransitiveFingerprint(ev_best_response)
    data = tf.fingerprint(turns=turns, progress_bar=False)
    tf.plot(ax=axes[1])

    filename = "img/Fingerprints/TransitiveFingerprint_index_%s" % index
    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":

    probe = axl.TitForTat
    step = 0.05
    points = axl.fingerprint._create_points(step=step)
    turns = 200
    num_process = 2

    for index, row in df.iterrows():

        best_response = axl.MemoryOnePlayer(
            [
                row["mem_one_p_1"],
                row["mem_one_p_2"],
                row["mem_one_p_3"],
                row["mem_one_p_4"],
            ]
        )
        ev_best_response = axl.MemoryOnePlayer(
            [
                row["evol_mem_one_p_1"],
                row["evol_mem_one_p_2"],
                row["evol_mem_one_p_3"],
                row["evol_mem_one_p_4"],
            ]
        )

        jobs = [
            dask.delayed(get_Ashlock_fingerprints)(
                best_response, ev_best_response, index
            ),
            dask.delayed(get_transitive_fingerprints)(
                best_response, ev_best_response, index
            ),
        ]

        dask.compute(*jobs, num_workers=num_process)
