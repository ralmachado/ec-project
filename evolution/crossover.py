import operator
import random


def arithmetical_cross(alpha):
    def _arithmetical_cross(p1: tuple, p2: tuple, prob_cross: float):
        if random.random() >= prob_cross:
            return p1, p2
        size = len(p1[0])
        cromo_1 = p1[0]
        cromo_2 = p2[0]
        f1 = [None] * size
        f2 = [None] * size
        for i in range(size):
            f1[i] = alpha * cromo_1[i] + (1 - alpha) * cromo_2[i]
            f2[i] = (1 - alpha) * cromo_1[i] + alpha * cromo_2[i]
        return (f1, 0), (f2, 0)

    return _arithmetical_cross


def heuristic_crossover(a: float):
    """
    Heuristic Crossover
    """

    def _heuristic_cross(p1: tuple, p2: tuple, prob: float, maximisation: bool = False):
        if random.random() >= prob:
            return p1, p2
        op = operator.gt if maximisation else operator.lt
        if op(p1[1], p2[1]):
            return [([a * (p1[0][i] - p2[0][i]) + p1[0][i] for i in range(len(p1[0]))], 0)]
        else:
            return [([a * (p2[0][i] - p1[0][i]) + p2[0][i] for i in range(len(p1[0]))], 0)]

    return _heuristic_cross
