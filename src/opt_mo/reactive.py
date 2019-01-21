"""
A script from identifying best responses of reactive strategies
"""

import itertools
from functools import partial

import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from sympy.polys import subresultants_qq_zz

import opt_mo


def round_matrix_expressions(matrix, num_digits, variable):
    """
    Rounds matrix elements. The elements are polynomials of a given variable.
    """
    for i, element in enumerate(matrix):
        matrix[i] = element.subs(
            {
                n: round(n, num_digits)
                for n in sym.Poly(element, variable).all_coeffs()
            }
        )
    return matrix


def eliminator_method(system, variable, other_variable):
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
    matrix = round_matrix_expressions(matrix, 8, variable)

    resultant = matrix.det()
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


def solve_system(system, variable, other_roots, other_variable):
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


def feasible_roots(coeffs):
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


def reactive_set(opponents):
    """
    Creates a set of possible optimal solutions.
    """
    p_1, p_2 = sym.symbols("p_1, p_2")
    utility = -opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents)

    derivatives = [sym.diff(utility, i) for i in [p_1, p_2]]
    derivatives = [expr.factor() for expr in derivatives]

    fractions = [sym.fraction(expr) for expr in derivatives]
    num = [expr[0] for expr in fractions]
    den = [expr[1] for expr in fractions]

    # roots p_1 for derivative
    p_one_roots = eliminator_method(num, p_1, p_2)

    p_two_roots = set()
    if p_one_roots:
        p_two_roots.update(solve_system(num, p_2, p_one_roots, p_1))

    # roots of p_2 for edges
    for p_one_edge in [0, 1]:
        coeffs = sym.Poly(num[1].subs({p_1: p_one_edge}), p_2).all_coeffs()
        roots = feasible_roots(coeffs)
        p_two_roots.update(roots)

    solution_set = p_one_roots | p_two_roots | set([0, 1])
    return solution_set


def argmax(opponents, solution_set):
    solutions = [
        (p_1, p_2, -opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents))
        for p_1, p_2 in itertools.product(solution_set, repeat=2)
    ]
    return max(solutions, key=lambda item: item[-1])


def reactive_best_response(opponents):
    """
    Calculates the best response reactive strategy using resultant theory.
    """
    solution_set = reactive_set(opponents)
    solution = argmax(opponents=opponents, solution_set=solution_set)

    return (solution[0], solution[1], solution[0], solution[1])


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
