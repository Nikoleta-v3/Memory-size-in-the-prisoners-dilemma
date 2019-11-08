"""
A script from identifying best responses of reactive strategies
"""

import itertools
from functools import partial

import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np
import skopt
import sympy as sym
from sympy.polys import subresultants_qq_zz

import opt_mo


def _roots_using_eliminator_method(system, variable, other_variable):
    """
    For solving a 2 polynomial system of 2 unknowns. Calculates the real roots
    of the first unknown using Sylvester's resultant or the eliminator.

    Note that there is a constrain that the roots must be between 0 and 1.

    Attributes
    ----------
    system: list
        A list of size 2 with the polynomials
    variable: symbol
        The variable for which we are returning the roots
    other_variable: symbol
        The second variable of the system
    """
    matrix = subresultants_qq_zz.sylvester(system[0], system[1], other_variable)
    matrix = sym.N(matrix, 9)

    resultant = matrix.det(method="berkowitz")
    num, den = sym.fraction(resultant.factor())

    # candidate roots
    coeffs = sym.Poly(num, variable).all_coeffs()
    roots = np.roots(coeffs)
    feasible_roots = set()
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.add(root)

    if den != 1:
        for root in feasible_roots:
            if den.subs({variable: root}) == 0:
                feasible_roots.remove(root)

    return feasible_roots


def _roots_solving_system_of_singel_unknown(
    system, variable, other_roots, other_variable
):
    """
    Solving the system for the second unknown once the roots of the first one
    have been calculated.

    Note that there is a constrain that the roots must be between 0 and 1.

    Attributes
    ----------
    system: list
        A list of size 2 with the polynomials
    variable: symbol
        The variable for which we are returning the roots
    other_roots: list
        A list with the roots of the first unknown
    other_variable: symbol
        The first unknown. The one that the roots have been calculated
    """
    roots = set()
    for root in other_roots:
        first_poly_coeffs = sym.Poly(
            system[0].subs({other_variable: root}), variable
        ).all_coeffs()
        first_poly_roots = set(np.roots(first_poly_coeffs))

        second_poly_coeffs = sym.Poly(
            system[1].subs({other_variable: root}), variable
        ).all_coeffs()
        second_poly_roots = set(np.roots(second_poly_coeffs))

        roots.update(first_poly_roots.intersection(second_poly_roots))

    feasible_roots = set()
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.add(root)
    return feasible_roots


def _roots_in_bound(coeffs):
    """
    Checks if the roots are feasible. In this work only solutions in [0, 1] are
    feasible.
    """
    roots = set(np.roots(coeffs))

    feasible_roots = set()
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.add(root)
    return feasible_roots


def get_candinate_reactive_best_responses(opponents):
    """
    Creates a set of possible optimal solutions.
    """
    p_1, p_2 = sym.symbols("p_1, p_2")
    utility = opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents)

    derivatives = [sym.diff(utility, i) for i in [p_1, p_2]]
    derivatives = [expr.factor() for expr in derivatives]

    fractions = [sym.fraction(expr) for expr in derivatives]
    num, den = [[expr for expr in fraction] for fraction in zip(*fractions)]

    candinate_roots_p_one = _roots_using_eliminator_method(num, p_1, p_2)

    candinate_roots_p_two = set()
    if len(candinate_roots_p_one) > 0:
        candinate_roots_p_two.update(
            _roots_solving_system_of_singel_unknown(
                num, p_2, candinate_roots_p_one, p_1
            )
        )

    for p_one in [0, 1]:
        coeffs = sym.Poly(num[1].subs({p_1: p_one}), p_2).all_coeffs()
        roots = _roots_in_bound(coeffs)
        candinate_roots_p_two.update(roots)

    candinate_set = candinate_roots_p_one | candinate_roots_p_two | set([0, 1])
    return candinate_set


def get_argmax(opponents, solution_set):
    solutions = [
        (p_1, p_2, opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents))
        for p_1, p_2 in itertools.product(solution_set, repeat=2)
    ]
    return max(solutions, key=lambda item: item[-1])


def get_reactive_best_response(opponents):
    """
    Calculates the best response reactive strategy using resultant theory.
    """
    solution_set = get_candinate_reactive_best_responses(opponents)
    solution = get_argmax(opponents=opponents, solution_set=solution_set)

    return np.array([solution[0], solution[1], solution[0], solution[1]])


def plot_reactive_utility(
    opponents, best_response_player=False, filename=False
):
    p_1, p_2 = sym.symbols("p_1, p_2")
    p = (p_1, p_2, p_1, p_2)

    p_one, p_two = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
    utility = -opt_mo.tournament_utility(p, opponents)

    expr = sym.lambdify((p_1, p_2), utility.simplify())

    plt.figure()
    X, Y = np.meshgrid(p_one, p_two)
    Z = expr(X, Y)

    plot = plt.contourf(X, Y, Z), plt.colorbar()
    if best_response_player:
        plt.plot(
            best_response_player[0],
            best_response_player[1],
            marker="x",
            color="r",
            markersize=20,
            markeredgewidth=5,
        )
    plt.ylabel(r"$p_2$"), plt.xlabel(r"$p_1$")

    if filename:
        plt.savefig(filename, bbox_inches="tight")

    return plot


def prepare_reactive_objective_optimisation(opponents):
    objective = partial(reactive_utility, opponents=opponents)
    return (
        lambda x: -objective(x)
        if (np.isnan(objective(x)) == False and np.isinf(objective(x)) == False)
        else 100
    )


def reactive_utility(p, opponents):
    utility = opt_mo.tournament_utility((p[0], p[1], p[0], p[1]), opponents)

    return utility


def get_reactive_best_response_with_bayesian(
    opponents,
    n_random_starts=40,
    n_calls=60,
    tol=10 ** -5,
    convergence_switch=True,
):
    """
    Approximates the best response reactive strategy using bayesian optimisation.
    """
    bounds = [(0, 1.0) for _ in range(2)]
    objective = prepare_reactive_objective_optimisation(opponents=opponents)

    method_params = {"n_random_starts": n_random_starts, "n_calls": n_calls}
    default_calls = n_calls

    while (
        method_params["n_calls"] == default_calls
        or not opt_mo.objective_is_converged(values, tol=tol)
    ) and convergence_switch:
        result = skopt.gp_minimize(
            func=objective,
            dimensions=bounds,
            acq_func="EI",
            random_state=0,
            **method_params
        )

        values = result.func_vals
        method_params["n_calls"] += 20

    return np.array([result.x[0], result.x[1], result.x[0], result.x[1]])
