import numpy as np
import opt_mo


def get_repeat_cycle_and_length(history, tol=10 ** -5):
    """
    Find any repeat of last moves in history. Including repeats of just one element 
    (so this can be used to check simple convergence as well as "cyclic convergence").

    Parameters
    ==========

    history: any iterator of numpy arrays.
    """
    size = len(history)
    for cycle_size in range(1, int(size / 2) + 1):
        if np.allclose(
            history[-cycle_size:],
            history[-2 * cycle_size : -cycle_size],
            atol=tol,
        ):
            return history[-cycle_size:], len(history[-cycle_size:])
    return None, float("inf")


def get_evolutionary_best_response(
    opponents,
    best_response_function,
    tol=10 ** -5,
    initial=np.array([1, 1, 1, 1]),
    K=1,
):
    """
    This implements the best response dynamics algorithm (Algorithm 1) in the
    manuscript.

    Given a set of opponents it repeatedly finds the best response to the
    opponents including a `K` self interactions.

    In some cases a single strategy will not be found but a cycle will appear.
    The best response strategy is the strategy in the cycle that achieves the
    highest utility.

    In the edge case that the utilities of all the strategies in the cycle are
    NaN, the best response strategy is the first strategy in the cycle.
    """

    history = [initial]
    best_response = best_response_function(opponents + history * K)
    history.append(best_response)

    _, repeat_length = get_repeat_cycle_and_length(history, tol=tol)
    while repeat_length >= float("inf"):

        best_response = best_response_function(opponents + [history[-1]] * K)
        print("Next generation.")
        history.append(best_response)
        cycle, repeat_length = get_repeat_cycle_and_length(history, tol=tol)

    utilities = [
        opt_mo.tournament_utility(player, opponents + [player] * K)
        for player in cycle
    ]

    if all(np.isnan(utilities)):
        best_response = cycle[0]
    else:
        best_response = cycle[utilities.index(np.nanmax(utilities))]

    return best_response, history, repeat_length
