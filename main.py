import random
from operator import itemgetter

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

from evolution import optimization
from evolution.algorithm import Evolution


def arithmetical_crossover(alpha):
    def arithmetical_cross(indiv_1, indiv_2, prob_cross):
        if random.random() >= prob_cross:
            return None
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        size = len(indiv_1[0])
        f1 = [None] * size
        f2 = [None] * size
        for i in range(size):
            f1[i] = alpha * cromo_1[i] + (1 - alpha) * cromo_2[i]
            f2[i] = (1 - alpha) * cromo_1[i] + alpha * cromo_2[i]
        return f1, f2

    return arithmetical_cross


def heuristic_crossover(a: float):
    """
    Heuristic Crossover (minimization)
    """

    def heuristic_cross(p1: tuple, p2: tuple, prob: float):
        if random.random() >= prob:
            return p1, p2
        if p1[1] < p2[1]:
            return [([a * (p1[0][i] - p2[0][i]) + p1[0][i] for i in range(len(p1[0]))], 0)]
        else:
            return [([a * (p2[0][i] - p1[0][i]) + p2[0][i] for i in range(len(p1[0]))], 0)]

    return heuristic_cross


def uniform_mutation(genotype, prob, domain):
    for i in range(len(genotype)):
        if random.random() < prob:
            genotype[i] = random.uniform(*domain[i])
    return genotype


def gauss_mutation(sigma):
    def _gauss_mutation(genotype, prob, domain):
        for i in range(len(genotype)):
            if random.random() < prob:
                genotype[i] += random.gauss(0, sigma)
                if genotype[i] < domain[i][0]:
                    genotype[i] = domain[i][0]
                if genotype[i] > domain[i][1]:
                    genotype[i] = domain[i][1]
        return genotype

    return _gauss_mutation


# Choice 0 mutation
def choice_zero():
    def zero(t, genotype, sk, T, b):
        def delta(t, y):
            return y * random.random() * (1 - (t / T)) ** b
        genotype[0] = delta(t = t, y = sk - genotype[0])
        return genotype
    return zero


# Tournament Selection
def tour_sel(t_size):
    def tournament(pop, maximisation: bool = False):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = sorted(random.sample(pop, t_size), key=itemgetter(1), reverse=maximisation)[0]
            mate_pool.append(winner)
        return mate_pool

    return tournament


# Survivals: elitism
def sel_survivors_elite(elite):
    def elitism(parents, offspring, maximisation: bool = False):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=maximisation)
        parents.sort(key=itemgetter(1), reverse=maximisation)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism


if __name__ == '__main__':
    # Problem and evolution parameters
    runs = 30
    dimensions = 5
    params = {
        "max_gen": 100,
        "population_size": 200,
        "domain": [[-5.12, 5.12]] * dimensions,
        "prob_mutation": 0.05,
        "prob_crossover": 0.7,
        "crossover_fn": arithmetical_crossover(alpha=0.2),
        #"mutation_fn": gauss_mutation(sigma=0.2),
        "mutation_fn": choice_zero(),
        "parents_fn": tour_sel(3),
        "survivors_fn": sel_survivors_elite(0.1),
        "fitness_fn": optimization.rastrigin,
        "maximisation": True,
    }

    # Create performance metric storage
    avg_by_gen = np.empty((runs, params["max_gen"] + 1))
    best_by_gen = np.empty((runs, params["max_gen"] + 1))

    # Run the evolutionary algorithm
    for run in trange(runs):
        evolution = Evolution(**params)
        avg_by_gen[run, 0] = evolution.avg_fitness
        best_by_gen[run, 0] = evolution.best_fitness

        for gen in evolution:
            avg_by_gen[run, gen] = evolution.avg_fitness
            best_by_gen[run, gen] = evolution.best_fitness

    overall_avg_by_gen = np.mean(avg_by_gen, axis=0)
    overall_best_by_gen = np.mean(best_by_gen, axis=0)

    # Plot performance
    sns.set_theme()
    plt.plot(overall_avg_by_gen, label='Average')
    plt.plot(overall_best_by_gen, label='Best')
    plt.title(f"Rastrigin ({'maximisation' if params['maximisation'] else 'minimisation'})")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()
