import numpy as np


def get_repeat_length_in_history(history, tol=10 ** -5):
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
            return len(history[-cycle_size:])
    return float("inf")


def get_evolutionary_best_response(
    opponents,
    best_response_function,
    tol=10 ** -5,
    initial=np.array([1, 1, 1, 1]),
):

    history = [initial]
    best_response = best_response_function(opponents + history)
    history.append(best_response)

    repeat_length = get_repeat_length_in_history(history, tol=tol)
    while repeat_length >= float("inf"):

        best_response = best_response_function(opponents + [history[-1]])
        print("Next generation.")
        history.append(best_response)
        repeat_length = get_repeat_length_in_history(history, tol=tol)

    return best_response, history, repeat_length
