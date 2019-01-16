import numpy as np

import opt_mo


def test_simulate_match_with_defector():
    assert np.isclose(
        opt_mo.simulate_match((0, 0, 0, 0), (0, 0, 0, 0)), 1, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((0, 0, 0, 0), (1, 1, 1, 1)), 5, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((0, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)),
        3,
        atol=10 ** -2,
    )


def test_simulate_match_with_cooperator():
    assert np.isclose(
        opt_mo.simulate_match((1, 1, 1, 1), (0, 0, 0, 0)), 0, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((1, 1, 1, 1), (1, 1, 1, 1)), 3, atol=10 ** -2
    )
    assert np.isclose(
        opt_mo.simulate_match((1, 1, 1, 1), (0.5, 0.5, 0.5, 0.5)),
        1.5,
        atol=10 ** -2,
    )
