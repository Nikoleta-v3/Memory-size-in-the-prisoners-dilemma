"""
A script from identifying best responses of reactive strategies
"""

import imp
import itertools
import sys
import time
from functools import partial

import axelrod as axl
import numpy as np
import pandas as pd
import skopt
import sympy as sym
from scipy.optimize import fsolve
from sympy.polys import subresultants_qq_zz

import opt_mo

def prepare_objective_optimisation(opponents):
    objective = partial(reactive_utility, opponents=opponents)
    return objective

def reactive_utility(p, opponents):
    p_1, p_2 = p
    utility = opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents)

    return utility

def round_matrix_expressions(matrix, num_digits, variable):
    """
    Rounds matrix elements. The elements are polynomials of a given variable.
    """
    for i, element in enumerate(matrix):
        matrix[i] = element.subs({n : round(n, num_digits) 
                             for n in sym.Poly(element, variable).all_coeffs()})
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
        first_poly_coeffs = sym.Poly(system[0].subs({other_variable: root}), 
                                     variable).all_coeffs()
        first_poly_roots = set(np.roots(first_poly_coeffs))

        second_poly_coeffs = sym.Poly(system[1].subs({other_variable: root}), 
                                     variable).all_coeffs()
        second_poly_roots = set(np.roots(second_poly_coeffs))

        roots.update(first_poly_roots.intersection(second_poly_roots))

    feasible_roots = set()
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.add(root)
    return feasible_roots

def feasible_roots(coeffs):
    roots = set(np.roots(coeffs))

    feasible_roots = set()
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.add(root)
    return feasible_roots

def reactive_set(opponents):
    p_1, p_2 = sym.symbols('p_1, p_2')
    utility = -opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents)
    
    # derivatives
    derivatives = [sym.diff(utility, i) for i in [p_1, p_2]]
    derivatives = [expr.factor() for expr in derivatives]
    
    # numerator
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
           
    solutions = [(p_1, p_2, -opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents))
                 for p_1, p_2 in itertools.product(solution_set, repeat=2)]
    return max(solutions, key=lambda item:item[-1])

def get_columns(method_params):
    cols = ['index', '$q_1$', '$q_2$', '$q_3$', '$q_4$', r'$\bar{q}_1$', r'$\bar{q}_2$',
            r'$\bar{q}_3$', r'$\bar{q}_4$', '$p_1 ^ *$', '$p_2 ^ *$', '$u_q$',
             'Optimisation time', '$U_G$', 'Training time', r'$\bar{p}_1 ^ *$',
             r'$\bar{p}_2 ^ *$']

    method_cols = ['{}'.format(key) for key in method_params.keys()]

    return cols + method_cols

if __name__ == '__main__':
    index = int(sys.argv[1])
    types = sys.argv[2]

    location = '~/rsc/Memory-size-in-the-prisoners-dilemma/data/reactive/' + types

    i = (index - 1) * 100
    while i <= index * 100:
        axl.seed(i)
        filename =  location + '/bayesian_{}.csv'.format(i)
        main_op = [np.random.random(4)]

        dfs = []
        for starts, calls in [(10, 20), (20, 30), (20, 40), (20, 45), (20, 50)]:
            method_params = {'n_random_starts' : starts, 'n_calls': calls}
            cols = get_columns(method_params)
            row = [i]
            row += [q for q in main_op[0]]
            if types == 'matches':
                cols = cols[:5] + cols[9:]
                list_opponents = main_op

            if types == 'tournaments':
                axl.seed(i + 10000)
                other = [np.random.random(4)]
                row += [q for q in other[0]]

                list_opponents = main_op + other

            start_optimisation = time.clock()
            solution_set = reactive_set(list_opponents)
            p_1, p_2, utility = opt_mo.argmax(list_opponents, solution_set)

            row.append(p_1), row.append(p_2), row.append(utility)
            row.append(time.clock() - start_optimisation)

            start_training = time.clock()
            objective = prepare_objective_optimisation(opponents=list_opponents)
            result = skopt.gp_minimize(func=objective, dimensions=[(0, .99999), (0, .99999)],
                                    acq_func="EI", random_state=0, **method_params)

            row.append(-result.fun), row.append(time.clock() - start_training)
            for vector in result.x:
                row.append(vector)

            for value in method_params.values():
                row.append(value)

            dfs.append(pd.DataFrame([row], columns=cols))
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(filename)
        i += 1
