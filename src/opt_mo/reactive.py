"""
A script from identifying best responses of reactive strategies
"""

import itertools
import sys
import time

import axelrod as axl
import numpy as np
import pandas as pd
import sympy as sym
from scipy.optimize import fsolve
from sympy.polys import subresultants_qq_zz

import opt_mo


def round_matrix_expressions(matrix, num_digits, variable):
    """
    Rounds matrix elements. The elements are polynomials of a given variable.
    """
    for i, element in enumerate(matrix):
        matrix[i] = element.subs({n : round(n, num_digits) 
                             for n in sym.Poly(element, variable).all_coeffs()})
    return matrix

def get_roots_of_first_unknown(system, variable, other_variable):
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
    roots = list(np.roots(coeffs))
    
    feasible_roots = []
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.append(root)
    
    if den != 1:
        for root in feasible_roots:
            if den.subs({variable: root}) == 0:
                feasible_roots.remove(root)

    return set(feasible_roots) | set([0, 1])

def get_roots_of_second_unknown(system, variable, other_roots, other_variable):
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
    roots = []
    for root in other_roots:
        first_poly = sym.lambdify(variable, system[0].subs({other_variable: root}))
        second_poly = sym.lambdify(variable, system[1].subs({other_variable: root}))
        roots.append(fsolve(equations, x0=[0, 0], args=[first_poly, second_poly])[0])

    feasible_roots = []
    for root in roots:
        if not np.iscomplex(root):
            if root >= 0 and root <= 1:
                feasible_roots.append(root)
    return feasible_roots

def equations(solution, args):
    polynomial_one, polynomial_two = args
    value, _ = solution
    return (polynomial_one(value), polynomial_two(value))

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

    # roots for p_1
    p_one_roots = get_roots_of_first_unknown(num, p_1, p_2)
    
    # roots for p_2
    if p_one_roots is not None:
        solution_set = p_one_roots
        p_two_roots = get_roots_of_second_unknown(num, p_2, p_one_roots, p_1)
    
    if p_two_roots is not None:
        solution_set |= set(p_two_roots)
    return solution_set

def argmax(opponents, solution_set):
           
    solutions = [(p_1, p_2, -opt_mo.tournament_utility((p_1, p_2, p_1, p_2), opponents))
                 for p_1, p_2 in itertools.product(solution_set, repeat=2)]
    return max(solutions, key=lambda item:item[-1])


def get_columns():
    cols = ['index', '$q_1$', '$q_2$', '$q_3$', '$q_4$', 'p_1^*', 'p_2^*', '$u_q$',
            'Optimisation time', r'$\bar{q}_1$', r'$\bar{q}_2$', r'$\bar{q}_3$',
            r'$\bar{q}_4$', 'p_1^*T', 'p_2^*T', '$u_q$', 'Optimisation time']
    return cols

if __name__ == '__main__':
    index = int(sys.argv[1])

    location = '~/rsc/Memory-size-in-the-prisoners-dilemma/data/random_numerical_experiments/'

    i = (index - 1) * 100
    while i <= index:
        axl.seed(i)
        filename =  location + 'reactive/{}.csv'.format(i)

        row = [i]
        main_op = [np.random.random(4)]
        row += [q for q in main_op[0]]

        # match
        start_optimisation = time.clock()

        solution_set = reactive_set(main_op)
        p_1, p_2, utility = opt_mo.argmax(main_op, solution_set)

        row.append(p_1), row.append(p_2), row.append(utility)
        row.append(time.clock() - start_optimisation)
        print('Finish match')
        print('------------')

        # tournament
        axl.seed(i + 10000)
        other = [np.random.random(4)]
        row += [q for q in other[0]]
        opponents = main_op + other

        print('Start Tournament')
        start_optimisation = time.clock()
        solution_set = reactive_set(opponents)
        p_1, p_2, u = opt_mo.argmax(opponents, solution_set)

        row.append(p_1), row.append(p_2), row.append(u)
        end_match = time.clock() - start_optimisation

        df = pd.DataFrame([row], columns=get_columns())
        df.to_csv(filename)
        i += 1
