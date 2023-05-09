"""
Common optimization problems for Evolutionary Algorithms
"""

import math
from random import uniform


def rastrigin(indiv):
    """
    rastrigin function
    domain = [-5.12, 5.12]
    minimum at (0,....,0)
    """
    n = len(indiv)
    A = 10
    return A * n + sum([x ** 2 - A * math.cos(2 * math.pi * x) for x in indiv])


def schwefel(indiv):
    """
    schwefel function
    domain = [-500; 500]
    minimum 0 at x = (420.9687,...,420.9687)
    """
    y = 418.9829 * len(indiv) - sum([x * math.sin(math.sqrt(math.fabs(x))) for x in indiv])
    return y


def quartic(indiv):
    """
    quartic = DeJong 4
    domain = [-1.28; 1.28]
    minimum 0 at x = 0
    """
    y = sum([(i + 1) * (x ** 4) for i, x in enumerate(indiv)])
    return y
