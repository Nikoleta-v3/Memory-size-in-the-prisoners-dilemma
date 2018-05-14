import numpy as np

def mem_quadratic_numerator(q):
    """
    Returns the matrix $Q$ for memory one strategies.
    """
    matrix = np.array([[0, -q[0] * q[1] + 5 * q[0] * q[3] + q[0] + q[1] * q[2] -
                        5 * q[2] * q[3] - q[2],
                        q[0] * q[2] - q[1] * q[2],
                        -5 * q[0] * q[2] + 5 * q[2] * q[3]],
                       [-q[0] * q[1] + 5 * q[0] * q[3] + q[0] + q[1] * q[2] -
                        5 * q[2] * q[3] - q[2], 0, q[0] * q[1] - q[0] * q[2] - 3
                        * q[1] * q[3] - q[1] + 3 * q[2] * q[3] + q[2],
                        5 * q[0] * q[2] - 5 * q[0] * q[3] - 3 * q[1] * q[2] +
                        3 * q[1] * q[3] - 2 * q[2] + 2 * q[3]],
                       [q[0] * q[2] - q[1] * q[2], q[0] * q[1] - q[0] * q[2] - 3
                        * q[1] * q[3] - q[1] + 3 * q[2] * q[3] + q[2],
                        0, 3 * q[1] * q[2] - 3 * q[2] * q[3]],
                       [-5 * q[0] * q[2] + 5 * q[2] * q[3], 5 * q[0] * q[2] - 5 *
                        q[0] * q[3] - 3 * q[1] * q[2] + 3 * q[1] * q[3] - 2 * q[2] +
                        2 * q[3], 3 * q[1] * q[2] - 3 * q[2] * q[3], 0]])
    return matrix

def mem_linear_numerator(q):
    """
    Returns the matrix $c$ for memory one strategies.
    """
    matrix = np.array([q[0] * q[1] - 5 * q[0] * q[3] - q[0],
                       -q[1] * q[2] + q[1] + 5 * q[2] * q[3] + q[2] - 5 * q[3] - 1,
                       -q[0] * q[1] + q[1] * q[2] + 3 * q[1] * q[3] + q[1] - q[2],
                        5 * q[0] * q[3] - 3 * q[1] * q[3] - 5 * q[2] * q[3] + 5 * q[2] - 2 * q[3]])
    return matrix

def mem_constant_numerator(q):
    """
    Returns the constant part $a$ for memory one strategies.
    """
    constant = - q[1] + 5 * q[3] + 1

    return constant

def mem_quadratic_denominator(q):
    """
    Returns the matrix $\bar{Q}$ for memory one strategies.
    """
    matrix = np.array([[0, -q[0] * q[1] + q[0] * q[3] + q[0] + q[1] * q[2] - q[2]
                        * q[3] - q[2], q[0] * q[2] - q[0] * q[3] - q[1] * q[2] +
                        q[1] * q[3], q[0] * q[1] - q[0] * q[2] - q[0] -
                        q[1] * q[3] + q[2] * q[3] + q[3]],
                       [-q[0] * q[1] + q[0]*q[3] + q[0] + q[1] * q[2] - q[2] * q[3]
                        - q[2], 0,  q[0] * q[1] - q[0] * q[2] - q[1] * q[3] - q[1]
                        + q[2] * q[3] + q[2],
                        q[0] * q[2] - q[0] * q[3] - q[1] * q[2] + q[1] * q[3]],
                       [q[0] * q[2] - q[0] * q[3] - q[1] * q[2] + q[1] * q[3],
                        q[0] * q[1] - q[0] * q[2] - q[1] * q[3] - q[1] + q[2] *
                        q[3] + q[2], 0, -q[0] * q[1] + q[0] * q[3] + q[1] * q[2]
                        + q[1] - q[2] * q[3] - q[3]],
                       [q[0] * q[1] - q[0] * q[2] - q[0] - q[1] * q[3] + q[2] * q[3]
                        + q[3], q[0] * q[2] - q[0] * q[3] - q[1] * q[2] + q[1] *
                        q[3], -q[0] * q[1] + q[0] * q[3] + q[1] * q[2] + q[1] -
                        q[2] * q[3] - q[3], 0]])
    return matrix

def mem_linear_denominator(q):
    """
    Returns the matrix $\bar{c}$ for memory one strategies.
    """
    matrix = np.array([q[0] * q[1] - q[0] * q[3] - q[0],
                     -q[1] * q[2] + q[1] + q[2] * q[3] + q[2] - q[3] - 1,
                     -q[0] * q[1] + q[1] * q[2] + q[1] - q[2] + q[3],
                      q[0] * q[3] - q[1] - q[2] * q[3] + q[2] - q[3] + 1])
    return matrix

def mem_constant_denominator(q):
    """
    Returns the constant part $\bar{a}$ for memory one strategies.
    """
    constant = -q[1] + q[3] + 1

    return constant

def utility(p, q):
    """
    Returns the utility of player p for a given opponent q.
    """
    x = np.array(p)

    Q = mem_quadratic_numerator(q)
    c = mem_linear_numerator(q)
    a = mem_constant_numerator(q)

    numerator = np.dot(x, Q.dot(x.T) * 1 / 2) + np.dot(c, x.T) + a

    Q_d = mem_quadratic_denominator(q)
    d = mem_linear_denominator(q)
    b = mem_constant_denominator(q)

    denominator = np.dot(x, Q_d.dot(x.T) * 1/2) + np.dot(d, x.T) + b

    return numerator / denominator

def tournament_utility(p, opponents):
    """
    Returns the utility of a player against a list of opponents.
    """
    obj = 0
    for q in opponents:
        obj += utility(p, q)
    return (obj / len(opponents))
