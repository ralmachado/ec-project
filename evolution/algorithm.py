"""
Simple Evolutionary Algorithm

Provides a modular, iterator-style, Evolution class, allowing for customizable performance
metrics computation as needed in the evolution loop.
"""

import random
from dataclasses import dataclass
from operator import itemgetter
from typing import Callable

import numpy as np


@dataclass
class Evolution:
    max_gen: int
    population_size: int
    domain: list[list[float]]
    prob_mutation: float
    prob_crossover: float
    crossover_fn: Callable
    mutation_fn: Callable
    parents_fn: Callable
    survivors_fn: Callable
    fitness_fn: Callable
    maximisation: bool = False

    def __post_init__(self):
        # Uniform population initialization
        self.population = [[random.uniform(*param) for param in self.domain] for _ in range(self.population_size)]
        self.population = [(individual, self.fitness_fn(individual)) for individual in self.population]
        self.gen = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.gen == self.max_gen:
            raise StopIteration

        self.gen += 1

        # Parent selection and Crossover
        mating_pool = self.parents_fn(self.population, maximisation=self.maximisation)
        parents = []
        for i in range(self.population_size - 1):
            x1 = mating_pool[i]
            x2 = mating_pool[i + 1]
            xover = self.crossover_fn(x1, x2, self.prob_crossover)
            parents.extend([(x, self.fitness_fn(x)) for x, _ in xover])
            if len(parents) == self.population_size:
                break

        # Mutation
        children = []
        for genotype, _ in parents:
            new_genotype = self.mutation_fn(genotype, self.prob_mutation, self.domain, t=self.gen, T=self.max_gen)
            children.append((new_genotype, self.fitness_fn(new_genotype)))
        # Survivor selection
        self.population = self.survivors_fn(self.population, children, maximisation=self.maximisation)
        self.population = [(x, self.fitness_fn(x)) for x, _ in self.population]
        return self.gen

    @property
    def avg_fitness(self):
        """
        Get current population's average fitness
        """
        return np.mean([fit for _, fit in self.population])

    @property
    def best_fitness(self):
        """
        Get current population's best fitness
        """
        self.population.sort(key=itemgetter(1), reverse=self.maximisation)
        return self.population[0][1]


def evolution(max_gen: int, population_size: int, domain: list[list[float]], prob_mutation: float,
              prob_crossover: float, crossover_fn: Callable, mutation_fn: Callable, parents_fn: Callable,
              survivors_fn: Callable, fitness_fn: Callable, maximisation: bool = False):
    # inicialize population: indiv = (cromo,fit)
    population = [([random.uniform(*param) for param in domain], 0) for _ in range(population_size)]
    # evaluate population
    population = [(indiv[0], fitness_fn(indiv[0])) for indiv in population]
    best = [best_pop(population)[1]]
    average = [np.mean([fit for _, fit in population])]
    for gen in range(max_gen):
        # sparents selection
        mate_pool = parents_fn(population)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(population_size - 1):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = crossover_fn(indiv_1, indiv_2, prob_crossover)
            progenitores.extend(filhos)
            # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation_fn(cromo, prob_mutation, domain)
            descendentes.append((novo_indiv, fitness_fn(novo_indiv)))
        # New population
        population = survivors_fn(population, descendentes)
        # Evaluate the new population
        population = [(indiv[0], fitness_fn(indiv[0])) for indiv in population]
        best.append(best_pop(population)[1])
        average.append(np.mean([fit for _, fit in population]))
    return best_pop(population), best, average


# auxiliary
def best_pop(populacao):
    """Minimization."""
    populacao.sort(key=itemgetter(1))
    return populacao[0]
